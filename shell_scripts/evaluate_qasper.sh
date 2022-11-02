
MODEL_DIR=$1
WEIGHTS_FILE=$2
QASPER_FILE=$3

allennlp evaluate $MODEL_DIR $QASPER_FILE  --include-package attribution --output-file tmp --cuda-device 0 --weights-file $WEIGHTS_FILE -o '{"validation_dataset_reader": {"type": "qasper_evidence_prompt", "model_name": "google/t5-xl-lm-adapt"}, "model": {"type": "seq2seq", "model_name": "google/t5-xl-lm-adapt"}, "data_loader": {"batch_size": 32}}' --predictions-output-file qasper_preds.jsonl #--file-friendly-logging

python scripts/evaluate_qasper_evidence_predictions.py --predictions qasper_preds.jsonl --data_file $QASPER_FILE