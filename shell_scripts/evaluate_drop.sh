
MODEL_DIR=$1
WEIGHTS_FILE=$2
DROP_GOLD=$3

# run allennlp evaluate. note that we change the model class, since drop is generative
allennlp evaluate $MODEL_DIR dummy.txt  --include-package attribution --output-file tmp --cuda-device 0 --weights-file $WEIGHTS_FILE -o '{"dataset_reader": {"type": "drop_reader", "model_name": "google/t5-base-lm-adapt", "split_name": "test"}, "model": {"type": "drop_model"}}' --predictions-output-file drop_preds.jsonl
# convert allennlp format to drop format
python scripts/convert_allennlp_pred_to_drop_eval_format.py drop_preds.jsonl
# run official drop script
python scripts/drop_eval_script.py --prediction_path drop_preds.json --gold_path $DROP_GOLD

echo "Remember to correct for the 1k examples missing (multiply score by 9535/8535~=1.11)"
