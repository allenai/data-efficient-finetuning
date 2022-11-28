local transformer_model = std.extVar('MODEL_NAME');
local epochs = 5;
local batch_size = 1;
local num_gradient_accumulation_steps = 8;
local num_gpus = 1;

# drop gets its own config as it uses a special model.
{
    "dataset_reader": {
        "type": "drop_reader",
	    "split_name": "train",
        "model_name": transformer_model,
    },
    "validation_dataset_reader": {
        "type": "drop_reader",
	    "split_name": "test",
        "model_name": transformer_model,
    },
    "train_data_path": "dummy",
    "validation_data_path": "dummy",
    "vocabulary": {
        "type": "empty",
    },
    "model": {
        "type": "drop_model",
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
        {"type": "should_validate_callback", "validation_interval": 1},
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
