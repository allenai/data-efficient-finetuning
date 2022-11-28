import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--predictions', type=str, required=True)
parser.add_argument('--data_file', type=str, required=True)
args = parser.parse_args()

references = {}
data = json.load(open(args.data_file))
for paper_data in data.values():
    for qa_info in paper_data["qas"]:
        qid = qa_info['question_id']
        all_evidence = []
        for answer_info in qa_info['answers']:
            all_evidence.append(answer_info['answer']['evidence'])
        references[qid] = all_evidence

predictions = {}
for line in open(args.predictions, 'r'):
    datum = json.loads(line)
    question_ids = [d["question_id"] for d in datum["metadata"]]
    paragraphs = [d["paragraph"] for d in datum["metadata"]]
    answer_choices = datum["metadata"][0]["answer_options"]
    response_indices = datum["response"]
    responses = [answer_choices[i] for i in response_indices]

    for qid, response, paragraph in zip(question_ids, responses, paragraphs):
        if qid not in predictions:
            predictions[qid] = []

        if "Yes" in response:
            predictions[qid].append(paragraph)

num_non_nulls = sum([p != [] for p in predictions.values()])
print(len([l for l in open(args.predictions, 'r')]))
print(f"Non null predictions: {num_non_nulls} / {len(predictions)} ({round(num_non_nulls / len(predictions) * 100, 2)}%)")

precision = 0
recall = 0
f1 = 0
base_precision = 0
base_recall = 0
base_f1 = 0

def compute_metrics(predictions, refs):
    if not refs:
        return (0.0, 0.0, 0.0) if predictions else (1.0, 1.0, 1.0)
    overlap = set(refs).intersection(predictions)
    precision = len(overlap) / len(predictions) if predictions else 1.0
    recall = len(overlap) / len(refs)
    f1 = (2*precision*recall / (precision + recall)) if (precision + recall) != 0 else 0.0
    return (precision, recall, f1)

for qid, q_references in references.items():
    metrics = [compute_metrics(predictions[qid], refs) for refs in q_references]
    max_precision, max_recall, max_f1 = sorted(metrics, key=lambda x: x[2], reverse=True)[0]
    precision += max_precision
    recall += max_recall
    f1 += max_f1
    baseline_metrics = [compute_metrics([], refs) for refs in q_references]
    max_b_precision, max_b_recall, max_b_f1 = sorted(baseline_metrics, key=lambda x: x[2], reverse=True)[0]
    base_precision += max_b_precision
    base_recall += max_b_recall
    base_f1 += max_b_f1

print(f"Precision: {precision/len(references)}")
print(f"Recall: {recall/len(references)}")
print(f"F1: {f1/len(references)}")

print("\nBaseline:")
print(f"Precision: {base_precision/len(references)}")
print(f"Recall: {base_recall/len(references)}")
print(f"F1: {base_f1/len(references)}")
