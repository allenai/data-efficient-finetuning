import sys
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


def print_dataset(dataset_generator, outfile_name):
    with open(outfile_name, "w") as outfile:
        for i, instance in enumerate(dataset_generator):
            if i >= 1000:
                break
            outfile.write(str(i) + "\t" + instance["pretokenized_input"].replace("\n", "").replace("\r", "").replace("\t", " ") + "\n")

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

for ds, reader in zip(ds_names, readers):
    print("printing out dataset: ", ds)
    print_dataset(reader.read('dummy'), f"queries/{ds}.tsv")

# qasper we handle special
print("printing out dataset: qasper")
qasper_reader = QasperEvidencePromptReader(return_original_query=True)
qasper_instances = [i for i in qasper_reader.read(qasper_file)]
random.shuffle(qasper_instances)
print_dataset(qasper_instances, "queries/qasper.tsv")
