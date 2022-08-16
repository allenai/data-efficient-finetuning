import faiss
import argparse
import torch
import numpy
import json
import tqdm
from scipy.stats import entropy
import pickle
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


numpy.random.seed(20389)


parser = argparse.ArgumentParser()
parser.add_argument("--search_output", type=str)
parser.add_argument("--training_data", type=str)
parser.add_argument("--encoded_training_data", type=str)
parser.add_argument("--index", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--encoding_batch_size", type=int)
parser.add_argument("--sorted_training_data", type=str)
parser.add_argument("--output_separate_files", action="store_true", help="write output after each iteration of kmeans++ ")
parser.add_argument("--training_data_distances", type=str)
parser.add_argument("--acquisition_batch_size", type=int, default=1000)
args = parser.parse_args()

dev_retrieved_indices = set()
for line in open(args.search_output):
    datum = json.loads(line)
    for id_ in datum["ids"]:
        dev_retrieved_indices.add(id_)


def encode_batch(batched_inputs):
    input_data = tokenizer.batch_encode_plus(batched_inputs,
                                             return_tensors="pt",
                                             padding=True)
    input_ids = input_data['input_ids']
    # (batch_size, num_tokens)
    mask = input_data['attention_mask']
    if torch.cuda.is_available():
        input_ids = input_ids.cuda(device=cuda_devices[0])
        mask = mask.cuda(device=cuda_devices[0])
    encoder_outputs = encoder(input_ids=input_ids,
                              attention_mask=mask,
                              return_dict=True)
    # (batch_size, num_tokens, hidden_size)
    hidden_states = encoder_outputs["last_hidden_state"]
    # (batch_size, hidden_size)
    pooled_hidden_states = (hidden_states * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
    return pooled_hidden_states.detach().cpu().numpy()

if os.path.exists(args.training_data_distances):
    print(f"Reading pickled distances")
    distance_data = pickle.load(open(args.training_data_distances, "rb"))
    training_data_distances = distance_data["training_data_distances"]
    training_distances_from_dev_retrieved = distance_data["training_dev_retrieved_distances"]
else:
    if os.path.exists(args.encoded_training_data):
        print(f"Reading encoded training data from {args.encoded_training_data}")
        training_data_matrix = pickle.load(open(args.encoded_training_data, "rb"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        assert torch.cuda.is_available()
        cuda_devices = list(range(torch.cuda.device_count()))
        print(f"Using CUDA devices {cuda_devices} for encoding training data")
        model.cuda(device=cuda_devices[0])
        model.eval()
        encoder = torch.nn.DataParallel(model.encoder, device_ids=cuda_devices)
        encoded_training_data = []
        with torch.inference_mode():
            batch = []
            for line in tqdm.tqdm(open(args.training_data)):
                instance = json.loads(line)
                batch.append(instance["input"])
                if len(batch) == args.encoding_batch_size:
                    encoded_batch = encode_batch(batch)
                    batch = []
                    encoded_training_data.append(encoded_batch)
            if batch:
                encoded_batch = encode_batch(batch)
                encoded_training_data.append(encoded_batch)

        training_data_matrix = numpy.concatenate(encoded_training_data)
        print(f"Dumping encoded training data at {args.encoded_training_data}")
        with open(args.encoded_training_data, "wb") as outfile:
            pickle.dump(training_data_matrix, outfile)

    print("Loading index..")
    index = faiss.read_index(args.index)
    print("Done loading index")
    opq_matrix = faiss.downcast_VectorTransform(index.chain.at(0))


    retrieved_data_matrix = numpy.asarray([index.reconstruct(i) for i in dev_retrieved_indices])
    # OPQ transform
    opq_training_data = opq_matrix.apply(training_data_matrix)
    opq_retrieved_data = opq_matrix.apply(retrieved_data_matrix)


    training_data_distances = faiss.pairwise_distances(opq_training_data, opq_training_data)
    training_retrieved_distances = faiss.pairwise_distances(opq_training_data, opq_retrieved_data)
    training_distances_from_dev_retrieved = training_retrieved_distances.min(axis=1)
    with open(args.training_data_distances, "wb") as outfile:
        pickle.dump(
                {
                    "training_data_distances": training_data_distances,
                    "training_dev_retrieved_distances": training_distances_from_dev_retrieved
                },
                outfile
                )

print("sorting training data using a KMeans++ like algorithm")

# We sort training data based on their distance to the retrieved set + the sorted set built so far.
# This is equivalent to running KMeans++ with the dev-retrieved set as a fixed set of centroids, and adding
# in-task training data as new centroids. Instead of stopping at a pre-determined number of cetroids, we
# sort the entire in-task training set while choosing a set of points (as many as args.acquisition_batch_size)
# at a time greedily.
sorted_training_indices = []
distances_to_selected = training_distances_from_dev_retrieved
print(f"Training to retrieved distances: {min(distances_to_selected)}, {numpy.mean(distances_to_selected)}, {max(distances_to_selected)}")
print(f"Training data distances: {numpy.min(training_data_distances)}, {numpy.mean(training_data_distances)}, {numpy.max(training_data_distances)}")
print(f"Number of negative training data distances: {numpy.sum(training_data_distances < 0.0)}")
print("Making all negative inter-training data distances 0")
training_data_distances = training_data_distances * (training_data_distances >= 0.0)

num_training_points = len(training_distances_from_dev_retrieved)
selection_pool = list(range(num_training_points))

if args.output_separate_files:
    outfile_prefix = args.sorted_training_data.replace(".jsonl", "")
else:
    outfile = open(args.sorted_training_data, "w")

training_data_lines = open(args.training_data).readlines()
with tqdm.tqdm(total=num_training_points) as pbar:
    set_index = 0
    while selection_pool:
        if args.output_separate_files:
            outfile = open(f"{outfile_prefix}_set{set_index}.jsonl", "w")
        if len(selection_pool) <= args.acquisition_batch_size:
            next_points = selection_pool
        else:
            distance_distribution = distances_to_selected / numpy.sum(distances_to_selected)

            uniform_entropy = entropy([1./len(selection_pool)] * len(selection_pool))
            print(f"Entropy: {entropy(distance_distribution)} (uniform: {uniform_entropy})")
            next_points = numpy.random.choice(
                    selection_pool,
                    args.acquisition_batch_size,
                    p=distance_distribution
                    )
            next_points = list(set(next_points))

        sorted_training_indices.extend(next_points)

        next_point_set = set(next_points)
        # Update distances and selection set
        next_distances = []
        next_pool = []
        for ind, distance in zip(selection_pool, distances_to_selected):
            # If the point is in the set of next points, we remove it from the selection pool.
            if ind not in next_point_set:
                next_pool.append(ind)
                distance = min(distance, min([training_data_distances[ind][j] for j in next_points]))
                next_distances.append(distance)

        selection_pool = next_pool
        distances_to_selected = numpy.asarray(next_distances)

        if args.output_separate_files:
            for ind in sorted_training_indices:
                print(training_data_lines[ind].strip(), file=outfile)

        set_index += 1
        pbar.update(len(next_points))


if not args.output_separate_files:
    for ind in sorted_training_indices:
        print(training_data_lines[ind].strip(), file=outfile)
