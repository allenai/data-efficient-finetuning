import json
import os
import tqdm
import gzip
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output_prefix", type=str, required=True)
parser.add_argument("--data_file", type=str, required=True)
parser.add_argument("--index_type", type=str, default="hnsw")
parser.add_argument("--max_batch_tokens", type=int, default=32000)
parser.add_argument("--add_interval", type=int, default=1000, help="Each index.add() will add add_interval points")
parser.add_argument("--write_interval", type=int, default=2000000, help="Each time after indexing roughly these many points the index will be written to disk")
parser.add_argument("--neighbors_per_node", type=int, default=512, help="HNSW parameter, default from DPR paper")
parser.add_argument("--construction_depth", type=int, default=200, help="HNSW parameter, default from DPR paper")
parser.add_argument("--search_depth", type=int, default=128, help="HNSW parameter, default from DPR paper")
parser.add_argument("--encoding_dim", type=int, default=512, help="Reduced dimensionality for OPQ")
parser.add_argument("--sq_train_size", type=int, default=1000000)
parser.add_argument("--device_ids", type=int, nargs="+")
args = parser.parse_args()

index_factory_string = f"OPQ8_{args.encoding_dim},HNSW{args.neighbors_per_node},PQ8"
index_prefix = f"p3_{args.model.replace('/', '-')}_{index_factory_string.replace(',', '-')}"

with open(os.path.join(args.output_prefix, f"{index_prefix}_hyperparameters.json"), "w") as outfile:
    json.dump(
            {
                "neighbors_per_node": args.neighbors_per_node,
                "construction_depth": args.construction_depth,
                "search_depth": args.search_depth,
                "encoding_dim": args.encoding_dim,
                "sq_train_size": args.sq_train_size
            }, outfile)

text_instances_file = args.data_file

assert os.path.exists(text_instances_file), "Text instances file does not exist!"

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
model.eval()
if torch.cuda.is_available():
    model.cuda(device=args.device_ids[0])
    num_gpus_available = len(args.device_ids)
    print(f"Using DataParallel for the encoder on {num_gpus_available} GPUs with ids {args.device_ids}")
    encoder = torch.nn.DataParallel(model.encoder, device_ids=args.device_ids)
else:
    encoder = model.encoder

if text_instances_file.endswith(".gz"):
    instances_file_ptr = gzip.open(text_instances_file, "rt")
else:
    instances_file_ptr = open(text_instances_file, "r")

def get_batches(num_instances_to_skip: int=0):
    batch = []
    num_batch_tokens = 0
    max_num_batch_tokens = args.max_batch_tokens
    num_batches_yielded = 0
    num_instances_yielded = 0
    num_truncated_instances = 0
    num_instances_read = 0
    started_batching = False
    while True:
        line = instances_file_ptr.readline()
        if not line:
            break
        num_instances_read += 1
        if num_instances_read <= num_instances_to_skip:
            continue
        if not started_batching:
            print(f"Starting to batch instances from instance number: {num_instances_read}")
            started_batching = True
        instance = json.loads(line)
        input_ = instance["input"]
        tokens = tokenizer.tokenize(input_)
        num_tokens = len(tokens)
        if num_tokens > tokenizer.max_len_single_sentence:
            num_truncated_instances += 1
        if num_tokens + num_batch_tokens < max_num_batch_tokens:
            batch.append(input_)
            num_batch_tokens += num_tokens
        else:
            yield batch
            num_instances_yielded += len(batch)
            num_batches_yielded += 1
            if num_batches_yielded % 10000 == 0:
                print(f"Average batch size: {num_instances_yielded / num_batches_yielded}")
                print(f"Truncated instances so far: {num_truncated_instances}")
            batch = [input_]
            num_batch_tokens = num_tokens
    if batch:
        yield batch
        print(f"Average batch size: {num_instances_yielded / num_batches_yielded}")
        print(f"Truncated instances so far: {num_truncated_instances}")

index_file = os.path.join(
        args.output_prefix,
        f"{index_prefix}.index"
)
index = None
last_written_index_size = 0
if os.path.exists(index_file):
    print(f"Index file exists. Reading {index_file}")
    index = faiss.read_index(index_file)
    last_written_index_size = index.ntotal
    print(f"Done reading index of size {last_written_index_size}")
else:
    print(f"Will write index to {index_file}")

aggregated_encoded_batches = []

print("Computing representations and indexing them")
with torch.inference_mode():
    for batch in tqdm.tqdm(get_batches(last_written_index_size)):
        input_data = tokenizer.batch_encode_plus(batch,
                                                 return_tensors="pt",
                                                 padding=True,
                                                 truncation=True)
        input_ids = input_data['input_ids']
        # (batch_size, num_tokens)
        mask = input_data['attention_mask']
        if torch.cuda.is_available():
            input_ids = input_ids.cuda(device=args.device_ids[0])
            mask = mask.cuda(device=args.device_ids[0])

        encoder_outputs = encoder(input_ids=input_ids,
                                  attention_mask=mask,
                                  return_dict=True)
        # (batch_size, num_tokens, hidden_size)
        hidden_states = encoder_outputs["last_hidden_state"]
        # (batch_size, hidden_size)
        pooled_hidden_states = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        pooled_hidden_states_np = pooled_hidden_states.detach().cpu().numpy()
        aggregated_encoded_batches.append(pooled_hidden_states_np)
        if index is None:
            hidden_size = pooled_hidden_states_np.shape[1]
            index = faiss.index_factory(hidden_size, index_factory_string)
            # We cannot access the HNSW parameters directly. `index` is of type IndexPreTransform. We need to downcast
            # the actual index to do this.
            hnswpq_index = faiss.downcast_index(index.index)
            hnswpq_index.hnsw.efConstruction = args.construction_depth
            hnswpq_index.hnsw.efSearch = args.search_depth

        if not index.is_trained and sum([x.shape[0] for x in aggregated_encoded_batches]) >= args.sq_train_size:
            print("Training index")
            data_to_train = numpy.concatenate(aggregated_encoded_batches)
            index.train(data_to_train)

        if index.is_trained and sum([x.shape[0] for x in aggregated_encoded_batches]) >= args.add_interval:
            data_to_add = numpy.concatenate(aggregated_encoded_batches)
            index.add(data_to_add)
            print(f"Added {data_to_add.shape[0]} points to index")
            aggregated_encoded_batches = []
            index_size = index.ntotal
            if index_size - last_written_index_size >= args.write_interval:
                print(f"Writing index of size {index_size}")
                faiss.write_index(index, index_file)
                last_written_index_size = index_size

    if aggregated_encoded_batches:
        if not index.is_trained:
            print("Training index")
            data_to_train = numpy.concatenate(aggregated_encoded_batches)
            index.train(data_to_train)
        data_to_add = numpy.concatenate(aggregated_encoded_batches)
        index.add(data_to_add)
        print(f"Added {data_to_add.shape[0]} points to index")

    faiss.write_index(index, index_file)
