from tqdm import tqdm
from datasets import load_dataset
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--p3_output_file", type=str, required=True)
args = parser.parse_args()


tasks = open('data/t0_prompt_tasks.txt', 'r')

outputfile = open(args.p3_output_file, 'w')

for task in tqdm(tasks):
    ds = load_dataset("bigscience/P3", task.strip(), split="train")
    for sample in ds:
        outputfile.write(json.dumps({
            'input': sample['inputs_pretokenized'],
            'target': sample['targets_pretokenized']
        }) + '\n')

