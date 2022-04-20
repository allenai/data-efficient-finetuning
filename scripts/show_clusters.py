import pickle
from datasets import load_dataset
import os
import json
from collections import defaultdict
from tqdm import tqdm


cluster_indices_directory = "/net/nfs.cirrascale/allennlp/hamishi/t0_clusters/cluster_indices/"

clusters_text = defaultdict(list)
errors = []
for filename in tqdm(os.listdir(cluster_indices_directory)):
    fullpath = os.path.join(cluster_indices_directory, filename)
    cluster_id = filename.replace("cluster_", "").replace("_indices.pkl", "")
    if "test" in cluster_id:
        split = "validation"
    else:
        split = "train"
    cluster_data = pickle.load(open(fullpath, "rb"))
    for dataset_id, indices in cluster_data.items():
        try:
            dataset = load_dataset("bigscience/P3", dataset_id, split=split)
        except ValueError:
            errors.append({"cluster_id": cluster_id, "dataset_id": dataset_id, "split": split})
            continue
        indices_set = set(indices)
        for index, datum in enumerate(dataset):
            if index > max(indices):
                break
            if index in indices_set:
                datum = {
                    "input": dataset[index]["inputs_pretokenized"],
                    "target": dataset[index]["targets_pretokenized"],
                    "dataset": dataset_id,
                    "index": index
                }
                if "is_correct" in dataset[index]:
                    datum["is_correct"] = dataset[index]["is_correct"]
                clusters_text[cluster_id].append(datum)



with open("t0_cluster_data.json", "w") as outfile:
    json.dump(clusters_text, outfile, indent=2)

with open("t0_cluster_errors.json", "w") as outfile:
    json.dump(errors, outfile, indent=2)

print("Sizes:")
for cluster_id, cluster_data in clusters_text.items():
    print(f"{cluster_id}: {len(cluster_data)}")
