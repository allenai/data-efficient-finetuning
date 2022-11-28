import gzip
import json
import random
from tqdm import tqdm
import sys

file = sys.argv[1]
outfile = open(sys.argv[1][:-6] + '_index.txt', 'w')
# start by loading the file into memory
with open(file) as f:
    for sample in f:
        outfile.write(f'{json.loads(sample)["index_id"]}\n')
