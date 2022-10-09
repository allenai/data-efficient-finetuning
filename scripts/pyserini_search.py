from pyserini.search.lucene import LuceneSearcher
import sys
import json
import random

from attribution.huggingface_readers import (
    RTEReader,
    ANLIR1Reader,
    ANLIR2Reader,
    ANLIR3Reader,
    WiCReader,
    WSCReader,
    WinoGrandeReader,
    HellaSwagReader,
    COPAReader,
    CBReader,
    StoryClozeReader,
    CaseHOLDReader
)
from attribution.drop_reader import DropMReader
from attribution.qasper_reader import QasperEvidencePromptReader

# qasper is the only reader that requires a file
qasper_file = sys.argv[1]

num_neighbours = 500
max_query = 1000

ds_names = [
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
    "storycloze",
    "casehold",
    "drop",
]
readers = [
    RTEReader(split_name='train', use_val_split=False, return_original_instance=True),
    ANLIR1Reader(split_name='train', use_val_split=False, return_original_instance=True),
    ANLIR2Reader(split_name='train', use_val_split=False, return_original_instance=True),
    ANLIR3Reader(split_name='train', use_val_split=False, return_original_instance=True),
    WiCReader(split_name='train', use_val_split=False, return_original_instance=True),
    COPAReader(split_name='train', use_val_split=False, return_original_instance=True),
    WSCReader(split_name='train', use_val_split=False, return_original_instance=True),
    WinoGrandeReader(split_name='train', use_val_split=False, return_original_instance=True),
    HellaSwagReader(split_name='train', use_val_split=False, return_original_instance=True),
    CBReader(split_name='train', use_val_split=False, return_original_instance=True),
    StoryClozeReader(split_name='train', use_val_split=False, return_original_instance=True),
    CaseHOLDReader(split_name='validation', return_original_instance=True),
    DropMReader(split_name='validation', return_original_instance=True)
]

searcher = LuceneSearcher('pyserini_index')

def retrieve(dataset_generator):
    retrieved_idxs = set()
    for j, instance in enumerate(dataset_generator):
        if j >= 1000:
            break
        hits = searcher.search(instance['pretokenized_input'])
        for i, hit in enumerate(hits):
            if i >= num_neighbours:
                break
            retrieved_idxs.add(hit.docid)
    return list(retrieved_idxs)

idxes = {}
for ds, reader in zip(ds_names, readers):
    print("processing out dataset: ", ds)
    retrieved = retrieve(reader.read('dummy'))
    with open(f"queries/{ds}_indices.txt", "w") as w:
        w.writelines(retrieved)
    idxes[ds] = retrieved

# qasper we handle special
print("printing out dataset: qasper")
qasper_reader = QasperEvidencePromptReader(return_original_query=True)
qasper_instances = [i for i in qasper_reader.read(qasper_file)]
random.shuffle(qasper_instances)
retrieved = retrieve(qasper_instances)
with open(f"queries/qasper_indices.txt", "w") as w:
    w.writelines(retrieved)
idxes['qasper'] = retrieved


# print out retrieved data
outfiles = [open(f'queries/{name}_bm25.jsonl', 'w') for name in ds_names]
data_file = open('/net/nfs.cirrascale/allennlp/hamishi/index-data/t0_only/p3_t5_base_filtered_instances.jsonl', 'r')
for i, line in data_file:
    instance = json.loads(line)
    for j, ds in enumerate(ds_names):
        if i in idxes[ds]:
            outfiles[j].write(str(i) + '\n')
