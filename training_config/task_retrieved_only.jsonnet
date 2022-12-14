local transformer_model = 'google/t5-xl-lm-adapt';
local epochs = 5;
local batch_size = 1;
local num_gradient_accumulation_steps = 8;

local p3_data_path = "data/p3_data_with_options.json";

local num_gpus = 4;


{
    "dataset_reader": {
        "type": "p3_jsonl",
        "model_name": transformer_model,
        "max_answer_length": 256
    },
    "train_data_path": std.extVar('TRAIN_DATA_PATH'),
    "validation_dataset_reader": {
      "type": std.extVar('VALIDATION_DATASET_READER_NAME'),
      "model_name": transformer_model,
      "use_val_split": false,
      "split_name": "validation"
    },
    "validation_data_path": "dummy",
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "seq2seq",
        "model_name": transformer_model,
    },
    "data_loader": {
        "batch_size": batch_size,
    },
    "trainer": {
      "optimizer": {
        "type": "adam",
        "lr": 5e-5,
      },
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": epochs,
        "cut_frac": 0.1,
      },
      "callbacks": [
	{"type": "tensorboard"},
      ],
      "checkpointer": {
        "keep_most_recent_by_count": 1
      },
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "enable_default_callbacks": false,
      "use_amp": false,
      "cuda_device": 0
    },
}
