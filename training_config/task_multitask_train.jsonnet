local transformer_model = std.extVar('MODEL_NAME');
local epochs = 5;
local batch_size = 1;
local num_gradient_accumulation_steps = 8;

local p3_data_path = "data/p3_data_with_options.json";

local num_gpus = 1;

{
    "dataset_reader": {
        "type": "multitask",
        "readers" :{
          "p3_idx": {
            "type": "p3_jsonl",
            "model_name": transformer_model,
          },
          "task": {
             "type": std.extVar('TRAIN_DATASET_READER_NAME'),
	            "split_name": "train",
              "model_name": transformer_model,
          },
        },
       
    },
    "train_data_path": {
      "p3_idx": std.extVar('P3_TRAIN_DATA_PATH'),
      "task": "dummy.txt"  # task loads data from huggingface
    },
    "validation_dataset_reader": {
        "type": std.extVar('VALIDATION_DATASET_READER_NAME'),
        "model_name": transformer_model,
        "split_name": "test"  # validation was our retrieved data.
    },
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "seq2seq",
        "model_name": transformer_model,
    },
    "data_loader": {
        "type": "multitask",
        "scheduler": {
          "type": "roundrobin",
          "batch_size": batch_size
        },
        "instances_per_epoch": std.extVar('INSTANCES_PER_EPOCH'), # rte + p3
        "sampler": "proportional"
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
        "keep_most_recent_by_count": epochs
      },
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "enable_default_callbacks": false,
      "use_amp": false,
      "cuda_device": 0,
    },
}
