import torch
import json
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy
from sklearn.decomposition import PCA
from sklearn import mixture
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/t5-small-lm-adapt"
#model_name = "t5-small"
encoder_block_name = "encoder.block.7"
#encoder_block_name = "encoder.block.5"
max_num_weights = 2048
num_clusters = 15
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.cuda()
parameters_of_interest = []
for name, parameter in model.named_parameters():
    if name.startswith(encoder_block_name) or name.startswith("encoder.final_layer"):
        parameters_of_interest.append((name, parameter))


data = json.load(open("p3_data_simplified.json"))
#data = json.load(open("p3_data.json"))
cluster_dir = f"/home/pradeepd/data/p3_dev_{model_name.replace('/', '_')}_final_layer_gradient_clusters/"
if not os.path.exists(cluster_dir):
    os.makedirs(cluster_dir)

print(f"Computing gradients on {model_name}, and will write clusters to {cluster_dir}")
print(f"Computing gradients only on the block {encoder_block_name} and the final layer norm weight")
print(f"Will keep track of only the {max_num_weights} max gradients")

instances = []

for dataset_info in data.values():
    for value in dataset_info["validation"].values():
        value["split"] = "validation"
        instances.append(value)
    for value in dataset_info["train"].values():
        value["split"] = "train"
        instances.append(value)

max_indices_file = os.path.join(cluster_dir, "max_indices.pkl")
if os.path.exists(max_indices_file):
    print(f"Found max indices at {max_indices_file}")
    max_indices = pickle.load(open(max_indices_file, "rb"))
else:
    indices_counts = None
    print("Computing gradients, first pass")
    for instance in tqdm(instances):
        inputs = tokenizer.encode(instance["input"], return_tensors="pt").cuda()
        targets = tokenizer.encode(instance["target"], return_tensors="pt").cuda()
        model_outputs = model(input_ids=inputs, labels=targets, return_dict=True)
        loss = model_outputs['loss']
        loss.backward(inputs=[p for n, p in parameters_of_interest])

        gradients = torch.cat([p.grad.flatten() for _, p in parameters_of_interest]).detach().cpu().numpy()
        if indices_counts is None:
            indices_counts = numpy.zeros_like(gradients)

        indices_counts[numpy.argsort(gradients)[-max_num_weights:]] += 1
        model.zero_grad()

    max_indices = numpy.argsort(indices_counts)[-max_num_weights:]
    coverage = sum(indices_counts[max_indices]) / sum(indices_counts)
    print(f"Coverage: {coverage}")
    with open(max_indices_file, "wb") as outfile:
        pickle.dump(max_indices, outfile)

max_gradients_file = os.path.join(cluster_dir, "all_max_gradients.pkl")
if os.path.exists(max_gradients_file):
    print(f"Found max gradients at {max_gradients_file}")
    all_max_gradients = pickle.load(open(max_gradients_file, "rb"))
else:
    print("Computing gradients, second pass")
    all_max_gradients = numpy.zeros((len(instances), max_num_weights))
    i = 0
    for instance in tqdm(instances):
        inputs = tokenizer.encode(instance["input"], return_tensors="pt").cuda()
        targets = tokenizer.encode(instance["target"], return_tensors="pt").cuda()
        model_outputs = model(input_ids=inputs, labels=targets, return_dict=True)
        loss = model_outputs['loss']
        loss.backward(inputs=[p for n, p in parameters_of_interest])

        gradients = torch.cat([p.grad.flatten() for _, p in parameters_of_interest]).detach().cpu().numpy()
        all_max_gradients[i] = gradients[max_indices]
        i += 1
        model.zero_grad()

    with open(max_gradients_file, "wb") as outfile:
        pickle.dump(all_max_gradients, outfile)

print("Running PCA")
pca = PCA(n_components=50, random_state=0)

ld_indexed_data = pca.fit_transform(all_max_gradients)

print("Clustering")
gmm = mixture.GaussianMixture(
    n_components=num_clusters,
    covariance_type='full',
    max_iter=150,
    random_state=0
)

gmm = gmm.fit(ld_indexed_data)

cluster_labels = gmm.predict(ld_indexed_data)

cluster_distances = cosine_distances(gmm.means_)

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
