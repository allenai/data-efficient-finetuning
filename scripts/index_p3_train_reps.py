from datasets import load_from_disk
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
parser.add_argument("--datasets", type=str, help="Json file containing the list of P3 datasets to load")
parser.add_argument("--output_prefix", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--index_type", type=str, default="hnsw")
parser.add_argument("--add_interval", type=int, default=10, help="Each index.add() will add add_interval * batch_size points")
parser.add_argument("--write_interval", type=int, default=2000000, help="Each time after indexing roughly these many points the index will be written to disk")
parser.add_argument("--neighbors_per_node", type=int, default=512, help="HNSW parameter, default from DPR paper")
parser.add_argument("--construction_depth", type=int, default=200, help="HNSW parameter, default from DPR paper")
parser.add_argument("--search_depth", type=int, default=128, help="HNSW parameter, default from DPR paper")
args = parser.parse_args()

with open(os.path.join(args.output_prefix, "hyperparameters.json"), "w") as outfile:
    json.dump({"neighbors_per_node": args.neighbors_per_node, "construction_depth": args.construction_depth, "search_depth": args.search_depth}, outfile)

text_instances_file = os.path.join(args.output_prefix, "p3_train_instances.jsonl")

if not os.path.exists(text_instances_file):
    print("Text instances file does not exist. Let's make one.")
    assert args.datasets is not None, "Processed text instances file does not exist. Provide --datasets"
    data_cache = "/net/nfs.cirrascale/allennlp/zhaofengw/t0/data_cache"
    train_tasks_list = json.load(open(args.datasets))
    if not os.path.exists(args.output_prefix):
        os.makedirs(args.output_prefix)

    instances = []
    instances_datasets_info = []

    for dataset_name in train_tasks_list:
        data_path = os.path.join(data_cache, dataset_name)
        if not os.path.exists(data_path):
            print(f"{data_path} not found!")
            continue
        dataset = load_from_disk(data_path)
        for split_name in dataset.keys():
            for i, instance_info in enumerate(dataset[split_name]):
                instances.append({
                    "input": instance_info['inputs_pretokenized'],
                    "target": instance_info['targets_pretokenized']
                    })
                if "answer_choices" in instance_info:
                    instances[-1]["answer_choices"] = instance_info["answer_choices"]
                instances_datasets_info.append((dataset_name, split_name, i))

    with gzip.open(text_instances_file, "wb") as outfile:
        json.dump(instances, outfile, indent=2)

    with gzip.open(os.path.join(args.output_prefix, "p3_train_instances_dataset_indices.json"), "wb") as outfile:
        json.dump(instances_datasets_info, outfile, indent=2)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
model.eval()
model.cuda()
num_gpus_available = torch.cuda.device_count()
print(f"Using DataParallel for the encoder on {num_gpus_available} GPUs")
parallel_encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(num_gpus_available)))

if text_instances_file.endswith(".gz"):
    instances_file_ptr = gzip.open(text_instances_file, "rb")
else:
    instances_file_ptr = open(text_instances_file, "r")

def get_batches():
    batch = []
    while True:
        line = instances_file_ptr.readline()
        if not line:
            break
        batch.append(json.loads(line))
        if len(batch) == args.batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

index = None
index_file = os.path.join(args.output_prefix, "p3_train_final_layer_rep.index")
last_written_index_size = 0
aggregated_encoded_batches = []

print("Computing representations and indexing them")
with torch.inference_mode():
    for batch in tqdm.tqdm(get_batches()):
        input_data = tokenizer.batch_encode_plus([instance["input"] for instance in batch],
                                                 return_tensors="pt",
                                                 padding=True)
        input_ids = input_data['input_ids'].cuda()
        # (batch_size, num_tokens)
        mask = input_data['attention_mask'].cuda()

        encoder_outputs = parallel_encoder(input_ids=input_ids,
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
            index = faiss.IndexHNSWFlat(hidden_size, args.neighbors_per_node)
            index.hnsw.efConstruction = args.construction_depth
            index.hnsw.efSearch = args.search_depth

        if len(aggregated_encoded_batches) >= args.add_interval:
            index.add(numpy.concatenate(aggregated_encoded_batches))
            aggregated_encoded_batches = []
            index_size = index.ntotal
            if index_size - last_written_index_size >= args.write_interval:
                print(f"Writing index of size {index_size}")
                faiss.write_index(index, index_file)
                last_written_index_size = index_size

    if aggregated_encoded_batches:
        index.add(numpy.concatenate(aggregated_encoded_batches))

    faiss.write_index(index, index_file)
