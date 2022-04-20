import torch
import json
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#model_name = "google/t5-small-lm-adapt"
model_name = "t5-small"
num_clusters = 15
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.cuda()


data = json.load(open("p3_data_simplified.json"))
#data = json.load(open("p3_data.json"))
cluster_dir = f"/home/pradeepd/data/p3_dev_{model_name.replace('/', '_')}_final_layer_clusters/"
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

print("Computing representations")
all_representations = None
i = 0
for instance in tqdm(instances):
    inputs = tokenizer.encode(instance["input"], return_tensors="pt").cuda()
    targets = tokenizer.encode(instance["target"], return_tensors="pt").cuda()
    model_outputs = model(input_ids=inputs, labels=targets, return_dict=True)
    representation = model_outputs["encoder_last_hidden_state"].detach().cpu().numpy().mean(1)[0]
    if all_representations is None:
        all_representations = numpy.zeros((len(instances), representation.shape[0]))
    all_representations[i] = representation
    i += 1

with open(os.path.join(cluster_dir, "final_layer_representations.pkl"), "wb") as outfile:
    pickle.dump(all_representations, outfile)

print("Running PCA")
pca = PCA(n_components=50, random_state=0)

ld_indexed_data = pca.fit_transform(all_representations)

print("Clustering")
gmm = mixture.GaussianMixture(
    n_components=num_clusters,
    covariance_type='full',
    max_iter=150,
    random_state=0
)

gmm = gmm.fit(ld_indexed_data)

cluster_labels = gmm.predict(ld_indexed_data)

cluster_distances = euclidean_distances(gmm.means_)

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

with open(os.path.join(cluster_dir, "cluster_distances.pkl"), "wb") as outfile:
    pickle.dump(cluster_distances, outfile)
