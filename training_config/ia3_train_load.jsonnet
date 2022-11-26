local transformer_model = std.extVar('MODEL_NAME');
local epochs = 1;
local batch_size = 8;
local num_gradient_accumulation_steps = 1;

local p3_data_path = "data/p3_data_with_options.json";

local num_gpus = 1;


{
    "dataset_reader": {
        "type": "p3_jsonl",
        "model_name": transformer_model,
        "max_query_length": 256
    },
    "train_data_path": std.extVar('TRAIN_DATA_PATH'),
    "validation_data_path": "dummy",
    "validation_dataset_reader": {
      "type": std.extVar('VALIDATION_DATASET_READER_NAME'),
      "model_name": transformer_model,
      "split_name": "validation",
      "val_size": 1000,
      "use_val_split": false,
      "max_query_length": 256
    },
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "ia3_seq2seq_load",
        "model_name": transformer_model,
        "load_from_file": std.extVar('WEIGHTS_NAME')
    },
    "data_loader": {
        "batch_size": batch_size,
        "batches_per_epoch": 1000,
        "shuffle": true
    },
    "validation_data_loader": {
        "batch_size": batch_size,
    },
    "trainer": {
      "optimizer": {
        "type": "huggingface_adafactor",
        "lr": 3e-3,
        "relative_step": false,
        "weight_decay": 0,
        "scale_parameter": true,
        "warmup_init": false
      },
      "learning_rate_scheduler": {
        "type": "polynomial_decay",
        "warmup_steps": 60,
        "end_learning_rate": 0
      },
      "callbacks": [
	      {"type": "tensorboard"},
      ],
      "grad_norm": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "enable_default_callbacks": false,
      "use_amp": false,
      "cuda_device": 0,
    },
}
