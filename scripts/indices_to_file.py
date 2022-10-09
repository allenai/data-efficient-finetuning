import tqdm
import sys
import json

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


#infiles = [f'/net/nfs.cirrascale/allennlp/hamishi/test/multi-task-attribution/retrieve/1000q_2500n_fixed_pool_t5_base/{outfile}_1000q_2500n_t5_base_indices.txt' for outfile in datasets]
infiles = [f'/net/nfs.cirrascale/allennlp/hamishi/test/multi-task-attribution/queries/{ds}_idxes.txt' for ds in datasets]
diff_indices = [set([int(i) for i in open(file, 'r')]) for file in infiles]
print('indices read')
outfiles = [f'/net/nfs.cirrascale/allennlp/hamishi/test/multi-task-attribution/queries/{ds}_bm25.jsonl' for ds in datasets]
files = [open(o, "w") for o in outfiles]
for i, line in tqdm.tqdm(enumerate(open('/net/nfs.cirrascale/allennlp/hamishi/index-data/t0_only/p3_t5_base_filtered_instances.jsonl', 'r'))):
    for j, indices in enumerate(diff_indices):
        if str(i) in indices:
            instance = json.loads(line)
            instance["index_id"] = i
            print(json.dumps(instance), file=files[j])