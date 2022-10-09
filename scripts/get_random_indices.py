import gzip
import json
import random
from tqdm import tqdm

# fill in these values with yours.
outfiles = ['qasper']#'anli_r1', 'anli_r2', 'anli_r3', 'casehold', 'cb', 'copa', 'drop', 'hellaswag', 'rte', 'story_cloze', 'wic', 'winogrande', 'wsc']
indices  = [58019]#721247, 755358, 1215546, 70622, 214031, 175802, 193014, 296924, 1049514, 205667, 197293, 189911, 132661]
max_index = 102864371
files = [open(f'/net/nfs.cirrascale/allennlp/hamishi/test/multi-task-attribution/retrieve/1000q_2500n_t5_base_fixed_pool_rand/{o}_random.jsonl', "w") for o in outfiles]

# generate random indices
random_indices = []
for idx in tqdm(indices):
    cur_idxes = []
    used = set()
    while len(cur_idxes) < idx:
        rand_int = random.randrange(max_index)
        if rand_int not in used:
            cur_idxes.append(rand_int)
            used.add(rand_int)
    random_indices.append(set(cur_idxes))

p3_data = '/net/nfs.cirrascale/allennlp/hamishi/index-data/t0_only/tmp.jsonl'
for i, line in tqdm(enumerate(open(p3_data))):
    for j, idx_set in enumerate(random_indices):
        if i in idx_set:
            instance = json.loads(line)
            print(json.dumps(instance), file=files[j])
print("\nDone writing random data")
