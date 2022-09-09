"""
A messy version of the retrieval script for my HuggingfaceRankReaders
"""

import argparse
import faiss
import torch
import gzip
import json
import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attribution.huggingface_readers import CaseHOLDReader, RTEReader, DROPReader


parser = argparse.ArgumentParser()
parser.add_argument("--index", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--search_output", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_neighbors_search", type=int, default=100)
parser.add_argument("--p3_data", type=str, help="If provided, will write training data to `training_data`")
parser.add_argument("--training_data", type=str)
parser.add_argument("--num_neighbors_write", type=int, default=20)
parser.add_argument("--p3_dataset_indices", type=str, help="If provided, will compute P3 dataset stats")
args = parser.parse_args()

# 
neighbours_to_write = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
query_sizes = [100]
args.num_neighbors_search = 5000

#reader = DROPReader(model_name=args.model, split_name='train')
reader = CaseHOLDReader(model_name=args.model, split_name='validation')
#reader = RTEReader(model_name=args.model, split_name='validation')
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
if torch.cuda.is_available():
    model.cuda()
model.eval()
index = faiss.read_index(args.index)
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)

def query_index(queries):
    input_data = tokenizer.batch_encode_plus(queries,
                                             return_tensors="pt",
                                             padding=True)
    input_ids = input_data['input_ids']
    # (batch_size, num_tokens)
    mask = input_data['attention_mask']
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        mask = mask.cuda()
    encoder_outputs = model.encoder(input_ids=input_ids,
                                       attention_mask=mask,
                                       return_dict=True)
    # (batch_size, num_tokens, hidden_size)
    hidden_states = encoder_outputs["last_hidden_state"]
    # (batch_size, hidden_size)
    pooled_hidden_states = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
    pooled_hidden_states_np = pooled_hidden_states.detach().cpu().numpy()
    return index.search(pooled_hidden_states_np, k=args.num_neighbors_search)

g = lambda: defaultdict(int)
per_neighbour_indices_frequencies = defaultdict(g)
outputfile = open(args.search_output, "w")
batch = []
with torch.inference_mode():
    for instance_idx, instance in enumerate(tqdm.tqdm(reader.read('dummy.txt'))):
        batch.append({"query": instance["pretokenized_input"]})
        if len(batch) == args.batch_size:
            batch_distances, batch_indices = query_index([i["query"] for i in batch])
            for instance, distances, indices in zip(batch, batch_distances, batch_indices):
                ids = [int(id_) for id_ in indices]
                for neigh in neighbours_to_write:
                    for id_ in ids[:neigh]:
                        per_neighbour_indices_frequencies[neigh][id_] += 1
                distances = [float(distance) for distance in distances]
                datum = {"ids": ids, "distances": distances}
                print(json.dumps(datum), file=outputfile)
            outputfile.flush()
            batch = []

print("\nDone searching.")
max_index = 0
for indices_frequencies in per_neighbour_indices_frequencies.values():
    mx_i = max(indices_frequencies.keys())
    max_index = max(mx_i, max_index)

outfiles = [args.training_data + f'_100q_{nn}n.jsonl' for nn in neighbours_to_write]
files = [open(o, "w") for o in outfiles]
for i, line in enumerate(gzip.open(args.p3_data)):
    if i > max_index:
        break
    for j, (num_neighbours, indices_frequencies) in enumerate(per_neighbour_indices_frequencies.items()):
        if i in indices_frequencies:
            instance = json.loads(line)
            instance["index_id"] = i
            instance["attribution_frequency"] = indices_frequencies[i]
            print(json.dumps(instance), file=files[j])

print("\nDone writing training data")
