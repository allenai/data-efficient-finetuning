import argparse
from fileinput import filename
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
from attribution.huggingface_readers import (
    CaseHOLDReader,
    RTEReader,
    CBReader,
    HellaSwagReader,
    StoryClozeReader,
    WinoGrandeReader,
    WSCReader,
    COPAReader,
    WiCReader,
    ANLIR1Reader,
    ANLIR2Reader,
    ANLIR3Reader
)
from attribution.drop_reader import DropMReader
from attribution.qasper_reader import QasperEvidencePromptReader
from attribution.p3_jsonl_reader import P3ClusterReader


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

data_files = []
training_data = []
datasets = [
    "rte",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "wic",
    "copa",
    "wsc",
    "winogrande",
    "hellaswag",
    "cb",
    "story_cloze",
    "casehold",
    "drop"
]

for dataset in datasets:
    if dataset == "rte":
        shorthand = "rte/32_shot"
    elif dataset == "anli_r1":
        shorthand = "anli-r1/50_shot"
    elif dataset == "anli_r2":
        shorthand = "anli-r2/50_shot"
    elif dataset == "anli_r3":
        shorthand = "anli-r3/50_shot"
    elif dataset == "cb":
        shorthand = "cb/32_shot"
    elif dataset == "copa":
        shorthand = "copa/32_shot"
    elif dataset == "hellaswag":
        shorthand = "h-swag/20_shot"
    elif dataset == "story_cloze":
        shorthand = "storycloze/70_shot"
    elif dataset == "wic":
        shorthand = "wic/32_shot"
    elif dataset == "winogrande":
        shorthand = "winogrande/50_shot"
    else:
        shorthand = "wsc/32_shot"
    for seed in [0, 1, 32, 42, 1024]:
        data_files.append(f"data/few_shot/{shorthand}/{seed}_seed.jsonl")
        training_data.append(f"{dataset}_{seed}.jsonl")

# readers for each dataset.
# we use train splits for t0 tasks, custom splits for other.
readers = [
    RTEReader(model_name=args.model, split_name='train', use_val_split=False),
    ANLIR1Reader(model_name=args.model, split_name='train', use_val_split=False),
    ANLIR2Reader(model_name=args.model, split_name='train', use_val_split=False),
    ANLIR3Reader(model_name=args.model, split_name='train', use_val_split=False),
    WiCReader(model_name=args.model, split_name='train', use_val_split=False),
    COPAReader(model_name=args.model, split_name='train', use_val_split=False),
    WSCReader(model_name=args.model, split_name='train', use_val_split=False),
    WinoGrandeReader(model_name=args.model, split_name='train', use_val_split=False),
    HellaSwagReader(model_name=args.model, split_name='train', use_val_split=False),
    CBReader(model_name=args.model, split_name='train', use_val_split=False),
    StoryClozeReader(model_name=args.model, split_name='train', use_val_split=False),
    CaseHOLDReader(model_name=args.model, split_name='validation'),
    DropMReader(model_name=args.model, split_name='validation')
]
# load index once into ram.
index = faiss.read_index(args.index)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
if torch.cuda.is_available():
    model.cuda()
model.eval()

per_ds_per_neighbour_indices_frequencies = {d: {} for d in datasets}
max_index = 0
#for data_file, out_file, reader in zip(data_files,training_data):
for dataset, reader in zip(datasets, readers):
    print(f"Retreiving over {dataset}")
    neighbours_to_write = [500]
    query_size = 1000
    args.num_neighbors_search = 500

    #reader = StoryClozeReader(model_name=args.model, split_name='validation')
    #reader = CaseHOLDReader(model_name=args.model, split_name='validation')
    #reader = RTEReader(model_name=args.model, split_name='train')
    #reader = P3ClusterReader(model_name=args.model)
    #reader = DROPReader(model_name=args.model, split_name='validation')
    filename = 'dummy'
    
    
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
    per_ds_per_neighbour_indices_frequencies[dataset] = defaultdict(g)
    outputfile = open(args.search_output, "w")
    batch = []
    with torch.inference_mode():
        for instance_idx, instance in enumerate(tqdm.tqdm(reader.read(filename))):
            if instance_idx >= query_size:
                break
            batch.append({"query": instance["pretokenized_input"]})
            if len(batch) == args.batch_size:
                batch_distances, batch_indices = query_index([i["query"] for i in batch])
                for instance, distances, indices in zip(batch, batch_distances, batch_indices):
                    ids = [int(id_) for id_ in indices]
                    for neigh in neighbours_to_write:
                        for id_ in ids[:neigh]:
                            per_ds_per_neighbour_indices_frequencies[dataset][neigh][id_] += 1
                    distances = [float(distance) for distance in distances]
                    datum = {"ids": ids, "distances": distances}
                    print(json.dumps(datum), file=outputfile)
                outputfile.flush()
                batch = []

    print(f"\nDone searching for {dataset}.")
    for indices_frequencies in per_ds_per_neighbour_indices_frequencies[dataset].values():
        mx_i = max(indices_frequencies.keys())
        max_index = max(mx_i, max_index)

print("Done searching for all datasets. Now writing data...")
# pause in case i havent made the file yet.

# create to save
# import pickle
# indices = [per_ds_per_neighbour_indices_frequencies[dataset][neighbours_to_write[0]] for dataset in datasets]
# pickle.dump(indices, open('tmp_indices.pkl', 'w'))
# indices = pickle.load(open('tmp_indices.pkl', 'rb'))

outfiles = [f'/net/nfs.cirrascale/allennlp/hamishi/test/multi-task-attribution/retrieve/1000q_2500n_fixed_pool_t5_base/{outfile}_1000q_2500n_t5_base_indices.txt' for outfile in datasets]
files = [open(o, "w") for o in outfiles]
for i, line in tqdm.tqdm(enumerate(open(args.p3_data))):
    if i > max_index:
        break
    for j, dataset in enumerate(datasets):
        indices_frequencies = per_ds_per_neighbour_indices_frequencies[dataset][neighbours_to_write[0]]
        if str(i) in indices_frequencies:
            instance = json.loads(line)
            instance["index_id"] = i
            instance["attribution_frequency"] = indices_frequencies[str(i)]
            print(json.dumps(instance), file=files[j])

# for i, line in tqdm.tqdm(enumerate(open(args.p3_data))):
#     for j, (per_ds_per_neighbour_indices_frequencies, dataset) in enumerate(zip(indices, datasets)):
#         if i in per_ds_per_neighbour_indices_frequencies:
#             instance = json.loads(line)
#             instance["index_id"] = i
#             instance["attribution_frequency"] = per_ds_per_neighbour_indices_frequencies[i]
#             print(json.dumps(instance), file=files[j])


print(f"\n Done writing training data for all datsets")
