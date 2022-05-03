import torch
import json
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy
from scipy.spatial.distance import mahalanobis
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--num_clusters", type=int, default=15)
parser.add_argument("--clustering", type=str, choices=['gmm', 'agglomerative'], required=True)
parser.add_argument("--distance", type=str, default='cosine', choices=['cosine', 'euclidean'])
parser.add_argument("--representations", type=str)
parser.add_argument("--pca", action="store_true")
parser.add_argument("--num_pca_components", type=int, default=50)
parser.add_argument("--data", type=str, default="data/p3_data_simplified.json")
parser.add_argument("--output_prefix", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=10)
args = parser.parse_args()

model_name = args.model
num_clusters = args.num_clusters
data = json.load(open(args.data))

print(f"Model: {args.model}")
print(f"Num clusters: {args.num_clusters}")


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.cuda()

if args.clustering == "gmm":
    cluster_dir = f"{args.output_prefix}/p3_dev_{model_name.replace('/', '_')}_final_layer_gmm_clusters/"
else:
    cluster_dir = f"{args.output_prefix}/p3_dev_{model_name.replace('/', '_')}_final_layer_aggl_{args.distance}_clusters/"

print(f"Clusters directory: {cluster_dir}")

if not os.path.exists(cluster_dir):
    os.makedirs(cluster_dir)


instances = []

for dataset_info in data.values():
    for value in dataset_info["validation"].values():
        value["split"] = "validation"
        instances.append(value)
    for value in dataset_info["train"].values():
        value["split"] = "train"
        instances.append(value)

batches = []
i = 0
while i < len(instances):
    batches.append(instances[i:i+args.batch_size])
    i += args.batch_size

if args.representations is None:
    print("Computing representations")
    all_representations = None
    i = 0
    for batch in tqdm(batches):
        input_data = tokenizer.batch_encode_plus([instance["input"] for instance in batch],
                                                 return_tensors="pt",
                                                 padding=True)
        target_data = tokenizer.batch_encode_plus([instance["target"] for instance in batch],
                                                  return_tensors="pt",
                                                  padding=True)
        input_ids = input_data['input_ids'].cuda()
        labels = target_data['input_ids'].cuda()
        # (batch_size, num_tokens)
        mask = input_data['attention_mask'].cuda()
        model_outputs = model(input_ids=input_ids,
                              attention_mask=mask,
                              labels=labels,
                              return_dict=True)
        # (batch_size, num_tokens, hidden_size)
        hidden_states = model_outputs["encoder_last_hidden_state"]
        # (batch_size, hidden_size)
        pooled_hidden_states = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        for representation in pooled_hidden_states:
            representation = representation.detach().cpu().numpy()
            if all_representations is None:
                all_representations = numpy.zeros((len(instances), representation.shape[0]))
            all_representations[i] = representation
            i += 1

    with open(os.path.join(cluster_dir, "final_layer_representations.pkl"), "wb") as outfile:
        pickle.dump(all_representations, outfile)
else:
    all_representations = pickle.load(open(args.representations, "rb"))

if args.pca:
    print("Running PCA")
    pca = PCA(n_components=50, random_state=0)

    all_representations = pca.fit_transform(all_representations)

if args.clustering == "gmm":
    print("Clustering with Gaussian Mixture")
    gmm = mixture.GaussianMixture(
        n_components=num_clusters,
        covariance_type='full',
        max_iter=150,
        random_state=0
    )

    gmm = gmm.fit(all_representations)

    cluster_labels = gmm.predict(all_representations)

    with open(os.path.join(cluster_dir, "cluster_means.pkl"), "wb") as outfile:
        pickle.dump(gmm.means_, outfile)

    with open(os.path.join(cluster_dir, "cluster_covars.pkl"), "wb") as outfile:
        pickle.dump(gmm.covars_, outfile)

    cluster_distances = numpy.zeros((num_clusters, num_clusters))
    inverse_covariances = [numpy.linalg.inv(x) for x in gmm.covars_]
    for i in range(num_clusters):
        for j in range(num_clusters):
            cluster_distances[i][j] = mahalanobis(gmm.means_[i], gmm.means_[j], inverse_covariances[j])
else:
    print(f"Clustering with an Agglomerative clustering algorithm using {args.distance} distance")
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        affinity=args.distance,
        compute_distances=True,
    )
    clustering = clustering.fit(all_representations)
    cluster_distances = clustering.distances_
    cluster_labels = clustering.labels_

with open(os.path.join(cluster_dir, "cluster_distances.pkl"), "wb") as outfile:
    pickle.dump(cluster_distances, outfile)

cluster_counts = [0] * 15
for label in cluster_labels:
    cluster_counts[label] += 1

print("Cluster counts:", cluster_counts)

cluster_index_map = defaultdict(lambda: defaultdict(lambda: {'train': [], 'validation': []}))

for cluster_label, instance in zip(cluster_labels, instances):
    cluster_index_map[cluster_label][instance['dataset']][instance['split']].append(instance['index'])

for cluster_label, cluster_data in cluster_index_map.items():
    with open(os.path.join(cluster_dir, f"cluster_{cluster_label}_indices.pkl"), "wb") as outfile:
        pickle.dump(dict(cluster_data), outfile)
