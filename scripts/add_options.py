import json
import sys
import tqdm
from collections import defaultdict

full_data = json.load(open(sys.argv[1]))
input_data = json.load(open(sys.argv[2]))
output_file = sys.argv[3]

special_dataset_prefixes = ["hellaswag", "winogrande", "super_glue_copa"]

dataset_options_dict = defaultdict(set)
instances_options_dict = defaultdict(lambda: {"instances": [], "options": []})


def is_dataset_special(dataset_name):
    for prefix in special_dataset_prefixes:
        if dataset_name.startswith(prefix):
            return True
    return False


for dataset_name, dataset_info in full_data.items():
    for split in ["validation", "train"]:
        for instance_id, instance_info in dataset_info[split].items():
            if is_dataset_special(dataset_name):
                instances_options_dict[instance_info["input"]]["instances"].append((dataset_name, split, instance_id))
                instances_options_dict[instance_info["input"]]["options"].append(instance_info["target"])
            else:
                dataset_options_dict[dataset_name].add(instance_info["target"])

instance_options = {}
for options_info in instances_options_dict.values():
    for instance_id in options_info["instances"]:
        instance_options[instance_id] = options_info["options"]

dataset_options_dict = {k: list(v) for k, v in dataset_options_dict.items()}
print(f"Accumulated dataset options of size {len(dataset_options_dict)}")

for dataset_name, dataset_info in tqdm.tqdm(input_data.items()):
    for split in ["validation", "train"]:
        for instance_id, instance_info in dataset_info[split].items():
            if is_dataset_special(dataset_name):
                options = instance_options[(dataset_name, split, instance_id)]
            else:
                options = dataset_options_dict[dataset_name]
            instance_info["options"] = options

print("Added options to all instances")

with open("dataset_options.json", "w") as outfile:
    json.dump(dataset_options_dict, outfile, indent=2)

with open("instance_specific_options.jsonl", "w") as outfile:
    for k, v in instance_options.items():
        print(json.dumps({"dataset": k[0], "split": k[1], "instance_id": k[2], "options": v}), file=outfile)

with open(output_file, "w") as outfile:
    json.dump(input_data, outfile, indent=2)
