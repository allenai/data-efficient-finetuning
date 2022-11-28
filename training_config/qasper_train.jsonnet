local transformer_model = std.extVar('MODEL_NAME');
local epochs = 5;
local batch_size = 1;
local num_gradient_accumulation_steps = 8;

local p3_data_path = "data/p3_data_with_options.json";

local num_gpus = 1;


{
    "dataset_reader": {
        "type": "qasper_evidence_prompt",
        "model_name": transformer_model
    },
    "train_data_path": std.extVar('TRAIN_DATA_PATH'),
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "seq2seq",
        "model_name": transformer_model
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
    }
}
