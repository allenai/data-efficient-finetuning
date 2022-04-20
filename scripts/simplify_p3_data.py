import json
from collections import defaultdict

# We'll keep only one copy of instance of all the prompt-variants

data_sizes = {k: len(v['validation']) + len(v['train']) for k, v in json.load(open('p3_data.json')).items()}

simplified_p3_data = defaultdict(lambda: {"validation": {}, "train": {}})

num_all_instances = 0
num_kept_instances = 0
for grouped_instance_info in json.load(open("p3_prompt_grouped_data.json")).values():
    dataset_groups = defaultdict(list)
    for instance_info in grouped_instance_info:
        num_all_instances += 1
        dataset_groups[instance_info["input"]["dataset"]].append((instance_info["input"], instance_info["cluster_id"]))
    largest_dataset = None
    for key in dataset_groups:
        if largest_dataset is None or data_sizes[key] > data_sizes[largest_dataset]:
            largest_dataset = key

    for instance, cluster_id in dataset_groups[largest_dataset]:
        if instance["is_correct"]:
            simplified_p3_data[largest_dataset]["validation" if "test" in cluster_id else "train"][instance["index"]] = instance
            num_kept_instances += 1


print(f"Kept {num_kept_instances} of {num_all_instances}")

with open("p3_data_simplified.json", "w") as outfile:
    json.dump(simplified_p3_data, outfile, indent=2)
