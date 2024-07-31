#!/home/ramahuja/myenv310/bin/python3

import pickle
import os
import h5py
import argparse

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed


def get_it_layer(model_name: str):
    it_layer_dict = {
        "alexnet": "features.12",
        "resnet50": "layer3.2.bn1",
        "resnet101": "layer3.2.bn1",
        "vgg16": "features.30",
        "vgg19": "features.36",
        "inception_v3": "Mixed_7a.branch3x3_1.bn",
        "vit_b_16": "encoder.layers.encoder_layer_8.mlp",
        "vit_b_32": "encoder.layers.encoder_layer_8.mlp",
        "efficientnet_v2_s": "features.6.7.stochastic_depth",
        "resnet18": "layer4.0.relu",
    }
    return it_layer_dict[model_name]

def get_pool_layers(model_name: str):
    pool_layers_dict = { 
        
        "resnet101": ['maxpool', 'layer1.1.add', 'layer2.0.add', 'layer2.3.add', 'layer3.1.add', 'layer3.4.add', 'layer3.7.add', 'layer3.10.add', 'layer3.13.add', 'layer3.16.add'],
        
        "vgg16": [
            "x",
            "features.3",
            "features.7",
            "features.12",
            "features.16",
            "features.21",
            "features.25",
            "features.30",
            "classifier.1",
            "classifier.5",
        ],
        "vgg19":  ['features.4', 'features.9','features.18','features.27','features.36'],
        
        "inception_v3": ['maxpool1', 'maxpool2', 'Mixed_5b.avg_pool2d', 'Mixed_5c.avg_pool2d', 'Mixed_5d.avg_pool2d', 'Mixed_6b.avg_pool2d', 'Mixed_6c.avg_pool2d', 'Mixed_6d.avg_pool2d', 'Mixed_6e.avg_pool2d', 'AuxLogits.avg_pool2d', 'AuxLogits.adaptive_avg_pool2d', 'Mixed_7b.avg_pool2d', 'Mixed_7c.avg_pool2d'],
        
        "alexnet": [
            "x",
            "features.1",
            "features.3",
            "features.6",
            "features.8",
            "features.11",
            "features.12",
            "avgpool",
            "classifier.1",
            "classifier.3",
            "classifier.6",
        ],
        "vit_b_16": [
            "encoder.layers.encoder_layer_0.mlp",
            "encoder.layers.encoder_layer_1.mlp",
            "encoder.layers.encoder_layer_2.mlp",
            "encoder.layers.encoder_layer_3.mlp",
            "encoder.layers.encoder_layer_4.mlp",
            "encoder.layers.encoder_layer_6.mlp",
            "encoder.layers.encoder_layer_7.mlp",
            "encoder.layers.encoder_layer_8.mlp",
            "encoder.layers.encoder_layer_9.mlp",
            "encoder.layers.encoder_layer_11.mlp",
        ],
        "vit_b_32": [
            "encoder.layers.encoder_layer_0.mlp",
            "encoder.layers.encoder_layer_1.mlp",
            "encoder.layers.encoder_layer_2.mlp",
            "encoder.layers.encoder_layer_3.mlp",
            "encoder.layers.encoder_layer_4.mlp",
            "encoder.layers.encoder_layer_6.mlp",
            "encoder.layers.encoder_layer_7.mlp",
            "encoder.layers.encoder_layer_8.mlp",
            "encoder.layers.encoder_layer_9.mlp",
            "encoder.layers.encoder_layer_11.mlp",
        ],
        "resnet50": ['maxpool', 'layer1.0.add', 'layer1.2.add', 'layer2.0.add', 'layer2.2.add', 'layer3.0.downsample.0', 'layer3.1.add', 'layer3.3.add', 'layer3.5.add', 'layer4.0.add'],
        "resnet18": ['maxpool', 'layer1.0.add', 'layer1.1.add', 'layer2.0.add', 'layer2.1.add', 'layer3.0.add', 'layer3.1.add', 'layer4.0.add', 'layer4.1.add', 'avgpool'],
        
        "efficientnet_v2_s": ['features.1.0.add', 'features.2.2.add', 'features.3.2.add', 'features.4.2.add', 'features.4.5.add', 'features.5.3.add', 'features.5.6.add', 'features.6.1.add', 'features.6.4.add', 'features.6.7.add'],
    
        }
    return pool_layers_dict.get(model_name, [])

def get_model_name(job_id: int):
    id_to_model_name = {
        0: "alexnet",
        1: "resnet50",
        2: "resnet101",
        3: "vgg19",
        4: "vit_b_16",
        5: "vit_b_32",
        6: "efficientnet_v2_s",
        7: "resnet18",
        8: "inception_v3",
    }
    return id_to_model_name[job_id]


def save_h5(feature_path, model_name, task_name, layer_name, dst_path):
    with open(os.path.join(feature_path, f"{model_name}_{task_name}_1.pkl"), "rb") as fin:
        features = pickle.load(fin)

    if layer_name in features:
        data = features[layer_name]
        trials = data.shape[0]
        data = data.reshape(trials, -1)
        print(f"Model: {model_name}, Task: {task_name}, Layer: {layer_name}, Shape: {data.shape}")
        with h5py.File(os.path.join(dst_path, f"{model_name}_{task_name}_{layer_name}.h5"), "w") as hf:
            hf.create_dataset("features", data=data)
    else:
        print(f"Layer {layer_name} not found in features for {model_name}")


def save_task(feature_path, model_name, task, dst_path):
    pool_layers = get_pool_layers(model_name)
    for layer in pool_layers:
        save_h5(feature_path, model_name, task, layer, dst_path)


def main(args):
    tasks = ["lamem", "lamem_shuffle"]
    model_names = args.model_names
    feature_path = args.src_path
    dst_path = args.dst_path
    os.makedirs(dst_path, exist_ok=True)

    if args.multiprocess:
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = []
            for model_name in model_names:
                for task in tasks:
                    futures.append(executor.submit(save_task, feature_path, model_name, task, dst_path))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                future.result()
    else:
        for model_name in model_names:
            for task in tasks:
                save_task(feature_path, model_name, task, dst_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pkl files to h5 files")
    parser.add_argument("--model_names", type=str, nargs="+", required=True, help="List of model names")
    parser.add_argument("--dst_path", type=str, required=True, help="Destination path for output files")
    parser.add_argument("--src_path", type=str, required=True, help="Source path for input files")
    parser.add_argument("--multiprocess", action='store_true', help="Enable multiprocessing")
    args = parser.parse_args()
    main(args)


# python load_feature_pkl_files_and_convert_to_h5.py --model_names resnet50 resnet101 alexnet vgg16 vgg19 inception_v3 --dst_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features_h5 --src_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features
