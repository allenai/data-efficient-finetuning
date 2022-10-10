
MODEL_DIR=$1
QASPER_FILE=$2
WEIGHTS_FILE=$3

allennlp evaluate $MODEL_DIR $QASPER_FILE  --include-package attribution --output-file tmp --cuda-device 0 --weights-file $WEIGHTS_FILE -o '{"dataset_reader": {"type": "qasper_evidence_prompt", "model_name": "google/t5-base-lm-adapt"}}' --predictions-output-file qasper_preds.jsonl

python scripts/evaluate_qasper_evidence_predictions.py --predictions qasper_preds.jsonl --data_file $QASPER_FILE
