import json
import sys

filename = sys.argv[1]
desired_size = int(sys.argv[2])
outfilename = sys.argv[3]

all_ids = []
all_distances = []

with open(filename, 'r') as f:
    for line in f:
        sample = json.loads(line)
        all_ids.append(sample['ids'])
        all_distances.append(sample['distances'])

def flatten_dedup_list(id_list):
    return list(set([i for sublist in id_list for i in sublist]))

cur_ret = 1000
cur_ids = flatten_dedup_list(all_ids)
# we trim ids off the end until we hit desired size
while len(cur_ids) > desired_size:
    all_ids = [i[:-1] for i in all_ids]
    cur_ids = flatten_dedup_list(all_ids)
    cur_ret -= 1
    print(f"Reduced down to {cur_ret} retrieved cur size {len(cur_ids)}", end='\r')

print(f'\nReached {len(cur_ids)} (as close to {desired_size} as poss) at ret {cur_ret}. Dumping now...')

# then save :)
with open(outfilename, 'w') as w:
    for i in cur_ids:
        w.write(str(i) + '\n')
