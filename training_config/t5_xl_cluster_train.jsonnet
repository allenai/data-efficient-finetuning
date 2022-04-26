local transformer_model = "google/t5-xl-lm-adapt";
local epochs = 5;
local batch_size = 1;
local num_gradient_accumulation_steps = 8;

local train_cluster_path = "/home/pradeepd/data/p3_dev_full_google_t5-xl-lm-adapt_final_layer_clusters/cluster_11_indices.pkl";
local dev_cluster_path = "/home/pradeepd/data/p3_dev_full_google_t5-xl-lm-adapt_final_layer_clusters/cluster_11_indices.pkl";
local p3_data_path = "data/p3_data_with_options.json";

local num_gpus = 1;


{
    "dataset_reader": {
        "type": "p3_cluster",
	"p3_data_path": p3_data_path,
	"split_name": "train",
        "model_name": transformer_model,
    },
    "validation_dataset_reader": {
        "type": "p3_cluster",
	"p3_data_path": p3_data_path,
	"split_name": "validation",
        "model_name": transformer_model,
    },
    "train_data_path": train_cluster_path,
    "validation_data_path": dev_cluster_path,
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
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "enable_default_callbacks": false,
      "use_amp": false,
      "cuda_device": 0,
    },
}
