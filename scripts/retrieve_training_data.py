import argparse
import faiss
import numpy
from sklearn.cluster import kmeans_plusplus
import torch
import json
import gzip
import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attribution.qasper_reader import QasperEvidencePromptReader


parser = argparse.ArgumentParser()
parser.add_argument("--dev_data", type=str)
parser.add_argument("--negative_sample_ratio", type=float, default=1.0)
parser.add_argument("--index", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--search_output", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_neighbors_search", type=int, default=500)
parser.add_argument("--p3_data", type=str, help="If provided, will write training data to `training_data`")
parser.add_argument("--training_data", type=str)
parser.add_argument("--num_neighbors_write", type=int, default=500)
parser.add_argument("--write_positive_neighbors_only", action="store_true", help="If set, will write neighbors of positive dev instances alone")
parser.add_argument("--coreset_size", type=int, default=None, help="If set, will use KMeans++ to select these many diverse points")
parser.add_argument("--p3_dataset_indices", type=str, help="If provided, will compute P3 dataset stats")
parser.add_argument("--stats_log", type=str, help="File to write the dataset stats")
parser.add_argument("--cuda_devices", type=int, nargs="+")
parser.add_argument("--retrieval_set_size", type=int, default=1000)
args = parser.parse_args()


indices_frequencies = defaultdict(int)
index = None
if not os.path.exists(args.search_output):
    assert args.dev_data is not None
    assert args.index is not None
    assert args.model is not None
    reader = QasperEvidencePromptReader(model_name=args.model, negative_sample_ratio=args.negative_sample_ratio)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    cuda_devices = [0]
    print(f"Using CUDA devices {cuda_devices}")
    if torch.cuda.is_available():
        model.cuda(device=cuda_devices[0])
    model.eval()
    encoder = torch.nn.DataParallel(model.encoder, device_ids=cuda_devices)
    print('loading index... (make take some time)')
    index = faiss.read_index(args.index)
    print('loaded index!')

    def query_index(queries):
        input_data = tokenizer.batch_encode_plus(queries,
                                                 return_tensors="pt",
                                                 padding=True)
        input_ids = input_data['input_ids']
        # (batch_size, num_tokens)
        mask = input_data['attention_mask']
        if torch.cuda.is_available():
            input_ids = input_ids.cuda(device=cuda_devices[0])
            mask = mask.cuda(device=cuda_devices[0])
        encoder_outputs = encoder(input_ids=input_ids,
                                  attention_mask=mask,
                                  return_dict=True)
        # (batch_size, num_tokens, hidden_size)
        hidden_states = encoder_outputs["last_hidden_state"]
        # (batch_size, hidden_size)
        pooled_hidden_states = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        pooled_hidden_states_np = pooled_hidden_states.detach().cpu().numpy()
        return index.search(pooled_hidden_states_np, k=args.num_neighbors_search)

    instances = [i for i in reader.read(args.dev_data)]
    # import random
    # random.shuffle(instances)
    # instances = instances[:args.retrieval_set_size]
    outputfile = open(args.search_output, "w")
    batch = []
    with torch.inference_mode():
        for instance in tqdm.tqdm(instances): #tqdm.tqdm(reader.read(args.dev_data)):
            metadata = instance.fields['metadata'].metadata
            batch.append({"question_id": metadata["question_id"], "query": metadata["query"], "paragraph_index": metadata["paragraph_index"], "target": metadata["target"]})
            if len(batch) == args.batch_size:
                batch_distances, batch_indices = query_index([i["query"] for i in batch])
                for instance_, distances, indices in zip(batch, batch_distances, batch_indices):
                    ids = [int(id_) for id_ in indices]
                    if not args.write_positive_neighbors_only or "Yes" in instance_["target"]:
                        for id_ in ids[:args.num_neighbors_write]:
                            indices_frequencies[id_] += 1
                    distances = [float(distance) for distance in distances]
                    datum = {"question_id": instance_["question_id"], "paragraph_index": instance_["paragraph_index"], "target": instance_["target"], "ids": ids, "distances": distances}
                    print(json.dumps(datum), file=outputfile)
                outputfile.flush()
                batch = []

    print("\nDone searching.")
else:
    print("Search output exists. Reading it instead of querying the index.")
    retrieved_data = [json.loads(line) for line in open(args.search_output)]
    for datum in retrieved_data:
        if not args.write_positive_neighbors_only or "Yes" in datum["target"]:
            for id_ in datum["ids"][:args.num_neighbors_write]:
                indices_frequencies[id_] += 1

if args.coreset_size is not None:
    print(f"Filtering down the retrieved training set to {args.coreset_size} points")
    if index is None:
        print("Loading index..")
        index = faiss.read_index(args.index)
        print("Done loading index")
    retrieved_indices = list(indices_frequencies.keys())
    # Inner index
    retrieved_vectors = numpy.asarray([index.index.reconstruct(i) for i in retrieved_indices])
    _, coreset_indices = kmeans_plusplus(retrieved_vectors, args.coreset_size)
    print("Finished running KMeans++")
    selected_indices = [retrieved_indices[i] for i in coreset_indices]
    indices_frequencies = {i: indices_frequencies[i] for i in selected_indices}


max_freq_indices = sorted(indices_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\nMost frequent indices: {max_freq_indices}")

max_index = max(indices_frequencies.keys())

if args.p3_data:
    with open(args.training_data, "w") as outfile:
        for i, line in tqdm.tqdm(enumerate(open(args.p3_data, "rt"))):
            if i > max_index:
                break
            if i in indices_frequencies:
                instance = json.loads(line)
                instance["index_id"] = i
                instance["attribution_frequency"] = indices_frequencies[i]
                print(json.dumps(instance), file=outfile)

print("\nDone writing training data")

if args.p3_dataset_indices:
    dataset_stats = defaultdict(lambda: {"seen": 0, "attributed": 0})
    for i, line in enumerate(gzip.open(args.p3_dataset_indices, "rt")):
        if i > max_index:
            break
        dataset_name, _, _ = line.strip().split("\t")
        dataset_stats[dataset_name]["seen"] += 1
        if i in indices_frequencies:
            dataset_stats[dataset_name]["attributed"] += 1

    num_all_seen = sum(x["seen"] for x in dataset_stats.values())
    num_all_attributed = sum(x["attributed"] for x in dataset_stats.values())
    stats = {}
    for d in dataset_stats:
        stats[d] = {"seen": dataset_stats[d]["seen"] / num_all_seen, "attributed": dataset_stats[d]["attributed"] / num_all_attributed}
    json.dump(stats, open(args.stats_log, "w"), indent=2)
