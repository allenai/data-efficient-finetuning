import json
import sys

stuff = {}

file = open(sys.argv[1], 'r')
for line in file:
    sample = json.loads(line)
    stuff[sample['query_id'][0]] = sample['answer'][0]

with open('drop_preds.json', 'w') as f:
    json.dump(stuff, f)
