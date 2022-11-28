
MODEL_DIR=$1
WEIGHTS_FILE=$2
NI_REF=natural_instructions/ni_reference.jsonl

# edit model to match your model size
# we re-use the drop reader for generation.
allennlp evaluate $MODEL_DIR dummy.txt  --include-package attribution --output-file tmp --cuda-device 0 --weights-file $WEIGHTS_FILE -o '{"validation_dataset_reader": {"type": "ni_reader", "model_name": "google/t5-xl-lm-adapt", "split_name": "test"}, "model": {"type": "ni_model", "model_name": "google/t5-xl-lm-adapt"}, "data_loader": {"batch_size": 1}}' --predictions-output-file ni_preds.jsonl
# run official drop script
python natural_instructions/ni_evaluation.py --prediction_file ni_preds.jsonl --reference_file $NI_REF

