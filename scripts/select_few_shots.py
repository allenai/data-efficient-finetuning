import faiss
import argparse
import torch
import numpy
import json
import tqdm
from scipy.stats import entropy
from sklearn.cluster import kmeans_plusplus
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


numpy.random.seed(20389)


parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, required=True)
parser.add_argument("--index", type=str, required=True)
parser.add_argument("--model", type=str)
parser.add_argument("--encoding_batch_size", type=int)
parser.add_argument("--selected_training_data", type=str)
parser.add_argument("--num_shots", type=int, default=32)
parser.add_argument("--apply_opq", action="store_true")
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
assert torch.cuda.is_available()
cuda_devices = list(range(torch.cuda.device_count()))
print(f"Using CUDA devices {cuda_devices} for encoding training data")
model.cuda(device=cuda_devices[0])
model.eval()
encoder = torch.nn.DataParallel(model.encoder, device_ids=cuda_devices)

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

raw_training_data = []
encoded_training_data = []
with torch.inference_mode():
    batch = []
    for line in tqdm.tqdm(open(args.training_data)):
        instance = json.loads(line)
        raw_training_data.append(instance)
        batch.append(instance["prompt_and_input"])
        if len(batch) == args.encoding_batch_size:
            encoded_batch = encode_batch(batch)
            batch = []
            encoded_training_data.append(encoded_batch)
    if batch:
        encoded_batch = encode_batch(batch)
        encoded_training_data.append(encoded_batch)

training_data_matrix = numpy.concatenate(encoded_training_data)


if args.apply_opq:
    print("Applying OPQ transform from the index")
    print("Loading index..")
    index = faiss.read_index(args.index)
    print("Done loading index")
    opq_matrix = faiss.downcast_VectorTransform(index.chain.at(0))


    training_data_matrix = opq_matrix.apply(training_data_matrix)

_, coreset_indices = kmeans_plusplus(training_data_matrix, args.num_shots)
selected_shots = [raw_training_data[i] for i in coreset_indices]

with open(args.selected_training_data, "w") as outfile:
    for instance in selected_shots:
        print(json.dumps(instance), file=outfile)