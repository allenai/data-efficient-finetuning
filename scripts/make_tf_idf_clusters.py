import json
import os
import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import mixture
from sklearn.metrics.pairwise import euclidean_distances

num_clusters = 15

data = json.load(open("p3_data_simplified.json"))
#data = json.load(open("p3_data.json"))
cluster_dir = "/home/pradeepd/data/p3_dev_tfidf_clusters/"
#cluster_dir = "/net/nfs.cirrascale/allennlp/pradeepd/p3_dev_full_tfidf_clusters/"

instances = []

for dataset_info in data.values():
    for value in dataset_info["validation"].values():
        value["split"] = "validation"
        instances.append(value)
    for value in dataset_info["train"].values():
        value["split"] = "train"
        instances.append(value)

vectorizer = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word', stop_words='english')
indexed_data = vectorizer.fit_transform([i["input"] for i in instances])

svd = TruncatedSVD(n_components=50, random_state=0)  # Cannot use PCA for sparse matrices. This is essentially LSA.

ld_indexed_data = svd.fit_transform(indexed_data)

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
