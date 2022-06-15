import argparse
import faiss
import torch
import json
import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attribution.qasper_reader import QasperEvidencePromptReader


parser = argparse.ArgumentParser()
parser.add_argument("--dev_data", type=str, required=True)
parser.add_argument("--negative_sample_ratio", type=float, default=1.0)
parser.add_argument("--index", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--search_output", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_neighbors_search", type=int, default=100)
parser.add_argument("--p3_data", type=str, help="If provided, will write training data to `training_data`")
parser.add_argument("--training_data", type=str)
parser.add_argument("--num_neighbors_write", type=int, default=20)
parser.add_argument("--p3_dataset_indices", type=str, help="If provided, will compute P3 dataset stats")
parser.add_argument("--stats_log", type=str, help="File to write the dataset stats")
args = parser.parse_args()


reader = QasperEvidencePromptReader(model_name=args.model, negative_sample_ratio=args.negative_sample_ratio)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
if torch.cuda.is_available():
    model.cuda()
model.eval()
index = faiss.read_index(args.index)

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


indices_frequencies = defaultdict(int)
outputfile = open(args.search_output, "w")
batch = []
with torch.inference_mode():
    for instance in tqdm.tqdm(reader.read(args.dev_data)):
        metadata = instance.fields['metadata'].metadata
        batch.append({"question_id": metadata["question_id"], "query": metadata["query"], "paragraph_index": metadata["paragraph_index"]})
        if len(batch) == args.batch_size:
            batch_distances, batch_indices = query_index([i["query"] for i in batch])
            for instance, distances, indices in zip(batch, batch_distances, batch_indices):
                ids = [int(id_) for id_ in indices]
                for id_ in ids[:args.num_neighbors_write]:
                    indices_frequencies[id_] += 1
                distances = [float(distance) for distance in distances]
                datum = {"question_id": instance["question_id"], "paragraph_index": instance["paragraph_index"], "ids": ids, "distances": distances}
                print(json.dumps(datum), file=outputfile)
            outputfile.flush()
            batch = []

print("\nDone searching.")
max_freq_indices = sorted(indices_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\nMost frequent indices: {max_freq_indices}")

max_index = max(indices_frequencies.keys())

if args.p3_data:
    with open(args.training_data, "w") as outfile:
        for i, line in enumerate(open(args.p3_data)):
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
    for i, line in enumerate(open(args.p3_dataset_indices)):
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
