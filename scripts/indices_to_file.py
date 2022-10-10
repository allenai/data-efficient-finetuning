import tqdm
import sys
import json
import argparse

datasets = [
    "rte",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "wic",
    "copa",
    "wsc",
    "winogrande",
    "hellaswag",
    "cb",
    "story_cloze",
    "casehold",
    "drop",
    "qasper"
]

parser = argparse.ArgumentParser()
parser.add_argument("--infile_path", type=str, required=True, help="directory containing index files")
parser.add_argument("--outfile_path", type=str, required=True, help="directory to output files")
parser.add_argument("--suffix", type=str, required=True, help="what to name the files with form \{dataset\}_\{suffix\}")
parser.add_argument("--p3_data", type=str, required=True, help="file containing the p3 data the index files reference")
args = parser.parse_args()


infile_path = args.infile_path
outfile_path = args.outfile_path
suffix = args.suffix
p3_data = args.p3_data

infiles = [f'{infile_path}/{ds}_idxes.txt' for ds in datasets]
diff_indices = [set([int(i) for i in open(file, 'r')]) for file in infiles]
print('indices read')
outfiles = [f'{outfile_path}/{ds}_{suffix}.jsonl' for ds in datasets]
files = [open(o, "w") for o in outfiles]
for i, line in tqdm.tqdm(enumerate(open(p3_data, 'r'))):
    for j, indices in enumerate(diff_indices):
        if str(i) in indices:
            instance = json.loads(line)
            instance["index_id"] = i
            print(json.dumps(instance), file=files[j])
