import json
from tqdm import tqdm
import sys

p3_instance_filename = sys.argv[1]
pyserini_output_filename = sys.argv[2]

with open(p3_instance_filename, 'r') as f:
    with open(pyserini_output_filename, "w") as writefile:
        for i, line in tqdm(enumerate(f)):
            data = json.loads(line)
            pyserini_sample = {
                "id": i,
                "contents": data["input"],
            }
            writefile.write(json.dumps(pyserini_sample) + "\n")
