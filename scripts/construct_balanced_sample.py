"""
A script to construct balanced random p3 sets. We fully balance this and uniformly sample from the list of tasks,
then sample a random input from this task. Note that this does not fully match t0 training (which only lightly balances
dataset sizes).
"""
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
import gzip
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--p3_data_file", type=str, required=True)
parser.add_argument("--p3_attribution_file", type=str, required=True)
parser.add_argument("--p3_dataset_mapping_file", type=str, required=True)
parser.add_argument("--num_samples", default=10000)
parser.add_argument('--output_folder', default='retrieve/rand_balanced')
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

rand = random.Random(args.seed)

p3_data = gzip.open(args.p3_data_file, mode='r')
p3_attribution = gzip.open(args.p3_attribution_file, 'r')
mapping = json.load(open(args.p3_dataset_mapping_file, 'r'))

num_samples = []
dataset_to_idxs = defaultdict(list)
all_datasets = list(mapping.keys())

for idx, line in p3_attribution:
    dataset_and_prompt, _, _ = line.strip().split('\t')
    dataset_to_idxs[dataset_and_prompt].append(idx)

samples_to_idxes = defaultdict(set)
for num_sample in num_samples:
    idxes = set()
    for i in range(0, num_sample):
        # randomly sample dataset
        dataset = rand.choice(all_datasets)
        # randomly sample prompt
        dataset_and_prompt = rand.choice(mapping[dataset])
        # randomly sample idx
        instance_idx = rand.choice(dataset_to_idxs[dataset_and_prompt])
        idxes.add(instance_idx)
    samples_to_idxes[num_sample] = idxes

# dump everything out
outfiles = [open(f'{n}_rand.json' for n in num_samples)]
for idx, sample in enumerate(p3_data):
    for j, n in enumerate(num_samples):
        if idx in num_samples[n]:
            instance = json.loads(sample)
            instance["index_id"] = i
            print(json.dumps(instance), file=outfiles[j])
