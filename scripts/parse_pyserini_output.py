import sys

# assumed format (from pyserini)
# 0 Q0 41840491 10 51.878502 Anserini

def parse_example(line):
    example = line.split(" ")
    # get the rank and the index
    rank = int(example[3])
    index = example[2]
    return rank, index

# filter the examples we keep
def filter_examples(examples, max_rank):
    return [(r, i) for (r, i) in examples if r < max_rank]

def construct_set_examples(examples):
    return list(set(i for _, i in examples))

output_file = open(sys.argv[1], 'r')

limit = int(sys.argv[3])

examples = [parse_example(l.strip()) for l in output_file]

max_rank = 500
num_samples = len(construct_set_examples(examples))

while num_samples > limit:
    max_rank -= 1
    nu_examples = filter_examples(examples, max_rank)
    index_list = construct_set_examples(nu_examples)
    nu_num_samples = len(index_list)
    # we will err to letting bm25 having more examples to be fair :)
    if nu_num_samples <= limit:
        print(f'\n\nWe have filtered down to size {num_samples} at max rank {max_rank+1} and finish\n')
        outfile = open(sys.argv[2], 'w')
        for idx in index_list:
            outfile.write(f'{idx}\n')
        outfile.close()
        break
    else:
        examples = nu_examples
        num_samples = nu_num_samples
    print(f'\rWe have filtered down to size {nu_num_samples} at max rank {max_rank}', end='')
