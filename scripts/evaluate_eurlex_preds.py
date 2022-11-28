import json
import sys
from transformers import T5TokenizerFast
import re
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

file = sys.argv[1]

tokenizer = T5TokenizerFast.from_pretrained('t5-base')
mlb = MultiLabelBinarizer()
concept_dict = json.load(open("data/eurovoc_descriptors.json", "r"))
eurlex_val = load_dataset('lex_glue', 'eurlex', split='validation')
labels = eurlex_val.features["labels"].feature.names
label_to_code = {concept_dict[l]['en']: l for l in labels}
total_labs = len(labels)
question_pattern = re.compile(r"Does the following text involve ([\w\s\-\,]+)\? Answer yes or no. Text: (.*).")

with open(file, 'r') as f:
    lines = f.readlines()
answers = []
last_text = 'im not in the first line'
cur_ans = []
cur_sample = -1
for i, line in enumerate(lines):
    result = json.loads(line)
    chosen_answer = tokenizer.decode(result['chosen_answer'][0], skip_special_tokens=True).strip().lower()
    chosen_answer_question = tokenizer.decode(result['chosen_answer_question'][0], skip_special_tokens=True)
    # extract the label we were asking about
    match = re.match(question_pattern, chosen_answer_question)
    term = match.group(1)
    text = match.group(2)
    if 'yes' in chosen_answer: 
        cur_ans.append(eurlex_val.features["labels"].feature.str2int(label_to_code[term]))
    # sometimes truncation can differ
    if last_text not in text and text not in last_text:
        answers.append(cur_ans)
        cur_ans = []
        cur_sample += 1
        #if text not in eurlex_val[cur_sample]['text'].replace('\n', ' '):
        #    import pdb; pdb.set_trace()
    last_text = text
            
 

preds = answers
true = eurlex_val['labels']
unlabelled_label = max([x for t in true for x in t])

# unfair-tos eval we add an extra 'unlabelled' label
for p in preds:
    if len(p) == 0:
        p.append(unlabelled_label)
for p in true:
    if len(p) == 0:
        p.append(unlabelled_label)

if len(preds) < len(true):
    print('huh?')
    while len(preds) < len(true):
        preds.append([])

mlb.fit(true + preds)
true = mlb.transform(true)
preds = mlb.transform(preds)
print('micro', f1_score(true, preds, average='micro'))
print('macro', f1_score(true, preds, average='macro'))
