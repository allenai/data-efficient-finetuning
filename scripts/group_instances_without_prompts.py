import json
from tqdm import tqdm
from collections import defaultdict

prompts = json.load(open("p3_prompts.json"))
cluster_data = json.load(open("./t0_cluster_data.json"))
raw_instances = defaultdict(list)
print("Grouping instances..")
prompt_issues_with_datasets = defaultdict(list)
for cluster_id, data in tqdm(cluster_data.items()):
    for instance in data:
        dataset = instance['dataset']
        prefix = prompts[dataset]['prefix']
        mids = prompts[dataset]['mid']
        suffix = prompts[dataset]['suffix']
        input_with_prompt = instance['input']
        original_input = instance['input'][len(prefix):]
        if len(suffix) != 0:
            original_input = original_input[:-len(suffix)]
        input_parts = []
        input_to_split = original_input
        for mid_part in mids:
            if mid_part == "":
                continue
            if mid_part not in input_to_split:
                prompt_issues_with_datasets[dataset].append(original_input)
                continue
            parts = input_to_split.split(mid_part, 1)
            split_input, rest = parts
            input_parts.append(split_input)
            input_to_split = rest
        input_parts.append(input_to_split)
        raw_instances[" ||| ".join(input_parts)].append({"cluster_id": cluster_id, "input": instance, "parts": input_parts})

with open('p3_prompt_grouping_errors.json', 'w') as outfile:
    json.dump(prompt_issues_with_datasets, outfile, indent=2)

print(f"Splitting issues in datasets: {prompt_issues_with_datasets.keys()}")

with open("p3_prompt_grouped_data.json", "w") as outfile:
    json.dump(raw_instances, outfile, indent=2)
