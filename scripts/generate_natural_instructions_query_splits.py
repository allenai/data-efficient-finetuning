import sys
import os
import json

from tqdm import tqdm

from attribution.ni_reader import NaturalInstructionsReader
from natural_instructions.task_eval_splits import ALL_EVAL_SPLITS

outfolder = sys.argv[1]

reader = NaturalInstructionsReader(
    return_original_instance=True,
    split_name='test',
    add_task_definition=False,
    num_pos_examples=0
)

split_queries = {k: [] for k in ALL_EVAL_SPLITS.keys() if 'cause' in k}

for instance in tqdm(reader.read('dummy')):
    for k, v in ALL_EVAL_SPLITS.items():
        if 'cause' not in k:
            continue
        if instance['preindexed_id'].split('-')[0] in v:
            split_queries[k].append({
                'input': instance['pretokenized_input'],
                'target': instance['pretokenized_target'],
                'id': instance['preindexed_id']
            })

for k, v in split_queries.items():
    with open(os.path.join(outfolder, f"{k}_queries.jsonl"), 'w') as w:
        for sample in tqdm(v):
            w.write(json.dumps(sample) + '\n')
