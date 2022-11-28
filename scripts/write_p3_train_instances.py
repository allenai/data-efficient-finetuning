import json
import gzip
import argparse
import os
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, help="Json file containing the list of P3 datasets to load")
parser.add_argument("--output_prefix", type=str, required=True)
parser.add_argument("--data_cache", type=str, required=True, help="location of the data cache")
parser.add_argument("--text_instances_file", type=str, required=True, help="output filename for processed data")

args = parser.parse_args()

data_cache = args.data_cache
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
