import json
from collections import defaultdict

data = json.load(open("./t0_cluster_data.json"))
cluster_stats = {}
dataset_to_clusters = defaultdict(list)

for cluster_id, cluster_data in data.items():
    input_set = set([x["input"] for x in cluster_data])
    dataset_set = set([x["dataset"] for x in cluster_data])
    cluster_stats[cluster_id] = {
            "size": len(input_set),
            "datasets": sorted(list(dataset_set))
    }
    for dataset in dataset_set:
        dataset_to_clusters[dataset].append(cluster_id)

with open("t0_cluster_stats.json", "w") as outfile:
    json.dump({"clusters": cluster_stats, "dataset_to_cluster": dataset_to_clusters}, outfile, indent=2)
