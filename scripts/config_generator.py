import yaml
import os

config_template = {
    "data": {
        "train_dataset": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/transform_lamem_256_train.ffcv",
        "val_dataset": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/transform_lamem_256_val.ffcv",
        "in_memory": 1,
        "num_workers": 30,
    },
    "dist": {"world_size": 4},
    "logging": {"folder": "./alexnet_logs", "log_level": 1},
    "lr": {
        "lr_schedule_type": "steplr",
        "lr_step_size": 30,
        "lr_gamma": 0.1,
        "lr_warmup_epochs": 0,
        "lr_warmup_method": "linear",
        "lr_warmup_decay": 0.01,
        "lr": 0.01,
    },
    "model": {"arch": "alexnet"},
    "resolution": {
        "end_ramp": 13,
        "max_res": 192,
        "min_res": 160,
        "start_ramp": 11,
        "fix_res": 0,
    },
    "training": {
        "task": "reg",
        "eval_only": 0,
        "batch_size": 128,
        "optimizer": "sgd",
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "epochs": 91,
        "label_smoothing": 0.1,
        "distributed": 0,
        "use_blurpool": 1,
        "bn_wd": 0,
    },
    "validation": {"batch_size": 512, "resolution": 256, "lr_tta": True},
}

batch_sizes = [32, 64, 128, 256, 512]
learning_rates = [0.01, 0.001, 0.0001, 0.00001]
weight_decays = [0.0001, 0.001, 0.01]

model = "alexnet_hp_configs"
output_path = os.path.join(
    "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster", model
)
os.makedirs(output_path, exist_ok=True)

for bs in batch_sizes:
    for lr in learning_rates:
        for wd in weight_decays:
            config = config_template.copy()
            config["training"]["batch_size"] = bs
            config["lr"]["lr"] = lr
            config["training"]["weight_decay"] = wd

            filename = os.path.join(output_path, f"config_bs{bs}_lr{lr}_wd{wd}.yaml")
            with open(filename, "w") as file:
                yaml.dump(config, file)
            print(f"Generated {filename}")
