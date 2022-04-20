import json
from tqdm import tqdm
from collections import defaultdict
import os

cluster_data = json.load(open("./t0_cluster_data.json"))

prompt_dataset_groups = defaultdict(list)
for instances in cluster_data.values():
    for instance in instances:
        prompt_dataset_groups[instance['dataset']].append(instance)

def infer_prefix_suffix(group):
    group_inputs = list(set([x['input'] for x in group]))
    instance1 = group_inputs[0]
    group_prefix = None
    group_suffix = None
    for instance2 in group_inputs[1:]:
        for i in range(1, len(instance1)):
            if instance1[:i] != instance2[:i]:
                prefix = instance1[:i-1]
                break
        if group_prefix is None:
            group_prefix = prefix
        else:
            group_prefix = prefix if len(prefix) < len(group_prefix) else group_prefix

        for i in range(1, len(instance1)):
            if instance1[-i:] != instance2[-i:]:
                suffix = instance1[-(i-1):] if i != 1 else ''
                break
        if group_suffix is None:
            group_suffix = suffix
        else:
            group_suffix = suffix if len(suffix) < len(group_suffix) else group_suffix

    return prefix, suffix

if os.path.exists("p3_prompts.json"):
    print("Prompts file exists!")
else:
    prompts = {}
    print("Inferring prompts..")
    for prompt_dataset_name, group in tqdm(prompt_dataset_groups.items()):
        prompt_prefix, prompt_suffix = infer_prefix_suffix(group)
        prompts[prompt_dataset_name] = {"prefix": prompt_prefix, "suffix": prompt_suffix}

    with open("p3_prompts.json", "w") as outfile:
        json.dump(prompts, outfile, indent=2)
