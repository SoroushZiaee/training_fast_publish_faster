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
        "alexnet": ['x', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'avgpool', 'flatten', 'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4', 'classifier.5', 'classifier.6'],
        "resnet50": ['x', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu_1', 'layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.downsample.0', 'layer1.0.downsample.1', 'layer1.0.add', 'layer1.0.relu_2', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu_1', 'layer1.1.conv3', 'layer1.1.bn3', 'layer1.1.add', 'layer1.1.relu_2', 'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.relu', 'layer1.2.conv2', 'layer1.2.bn2', 'layer1.2.relu_1', 'layer1.2.conv3', 'layer1.2.bn3', 'layer1.2.add', 'layer1.2.relu_2', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.relu_1', 'layer2.0.conv3', 'layer2.0.bn3', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.0.add', 'layer2.0.relu_2', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.relu_1', 'layer2.1.conv3', 'layer2.1.bn3', 'layer2.1.add', 'layer2.1.relu_2', 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.relu', 'layer2.2.conv2', 'layer2.2.bn2', 'layer2.2.relu_1', 'layer2.2.conv3', 'layer2.2.bn3', 'layer2.2.add', 'layer2.2.relu_2', 'layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.relu', 'layer2.3.conv2', 'layer2.3.bn2', 'layer2.3.relu_1', 'layer2.3.conv3', 'layer2.3.bn3', 'layer2.3.add', 'layer2.3.relu_2', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.relu_1', 'layer3.0.conv3', 'layer3.0.bn3', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.0.add', 'layer3.0.relu_2', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.relu_1', 'layer3.1.conv3', 'layer3.1.bn3', 'layer3.1.add', 'layer3.1.relu_2', 'layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.relu', 'layer3.2.conv2', 'layer3.2.bn2', 'layer3.2.relu_1', 'layer3.2.conv3', 'layer3.2.bn3', 'layer3.2.add', 'layer3.2.relu_2', 'layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.relu', 'layer3.3.conv2', 'layer3.3.bn2', 'layer3.3.relu_1', 'layer3.3.conv3', 'layer3.3.bn3', 'layer3.3.add', 'layer3.3.relu_2', 'layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.relu', 'layer3.4.conv2', 'layer3.4.bn2', 'layer3.4.relu_1', 'layer3.4.conv3', 'layer3.4.bn3', 'layer3.4.add', 'layer3.4.relu_2', 'layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.relu', 'layer3.5.conv2', 'layer3.5.bn2', 'layer3.5.relu_1', 'layer3.5.conv3', 'layer3.5.bn3', 'layer3.5.add', 'layer3.5.relu_2', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.relu_1', 'layer4.0.conv3', 'layer4.0.bn3', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.0.add', 'layer4.0.relu_2', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.relu_1', 'layer4.1.conv3', 'layer4.1.bn3', 'layer4.1.add', 'layer4.1.relu_2', 'layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.relu', 'layer4.2.conv2', 'layer4.2.bn2', 'layer4.2.relu_1', 'layer4.2.conv3', 'layer4.2.bn3', 'layer4.2.add', 'layer4.2.relu_2', 'avgpool', 'flatten', 'fc'],
        "resnet101": ['x', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu_1', 'layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.downsample.0', 'layer1.0.downsample.1', 'layer1.0.add', 'layer1.0.relu_2', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu_1', 'layer1.1.conv3', 'layer1.1.bn3', 'layer1.1.add', 'layer1.1.relu_2', 'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.relu', 'layer1.2.conv2', 'layer1.2.bn2', 'layer1.2.relu_1', 'layer1.2.conv3', 'layer1.2.bn3', 'layer1.2.add', 'layer1.2.relu_2', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.relu_1', 'layer2.0.conv3', 'layer2.0.bn3', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.0.add', 'layer2.0.relu_2', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.relu_1', 'layer2.1.conv3', 'layer2.1.bn3', 'layer2.1.add', 'layer2.1.relu_2', 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.relu', 'layer2.2.conv2', 'layer2.2.bn2', 'layer2.2.relu_1', 'layer2.2.conv3', 'layer2.2.bn3', 'layer2.2.add', 'layer2.2.relu_2', 'layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.relu', 'layer2.3.conv2', 'layer2.3.bn2', 'layer2.3.relu_1', 'layer2.3.conv3', 'layer2.3.bn3', 'layer2.3.add', 'layer2.3.relu_2', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.relu_1', 'layer3.0.conv3', 'layer3.0.bn3', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.0.add', 'layer3.0.relu_2', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.relu_1', 'layer3.1.conv3', 'layer3.1.bn3', 'layer3.1.add', 'layer3.1.relu_2', 'layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.relu', 'layer3.2.conv2', 'layer3.2.bn2', 'layer3.2.relu_1', 'layer3.2.conv3', 'layer3.2.bn3', 'layer3.2.add', 'layer3.2.relu_2', 'layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.relu', 'layer3.3.conv2', 'layer3.3.bn2', 'layer3.3.relu_1', 'layer3.3.conv3', 'layer3.3.bn3', 'layer3.3.add', 'layer3.3.relu_2', 'layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.relu', 'layer3.4.conv2', 'layer3.4.bn2', 'layer3.4.relu_1', 'layer3.4.conv3', 'layer3.4.bn3', 'layer3.4.add', 'layer3.4.relu_2', 'layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.relu', 'layer3.5.conv2', 'layer3.5.bn2', 'layer3.5.relu_1', 'layer3.5.conv3', 'layer3.5.bn3', 'layer3.5.add', 'layer3.5.relu_2', 'layer3.6.conv1', 'layer3.6.bn1', 'layer3.6.relu', 'layer3.6.conv2', 'layer3.6.bn2', 'layer3.6.relu_1', 'layer3.6.conv3', 'layer3.6.bn3', 'layer3.6.add', 'layer3.6.relu_2', 'layer3.7.conv1', 'layer3.7.bn1', 'layer3.7.relu', 'layer3.7.conv2', 'layer3.7.bn2', 'layer3.7.relu_1', 'layer3.7.conv3', 'layer3.7.bn3', 'layer3.7.add', 'layer3.7.relu_2', 'layer3.8.conv1', 'layer3.8.bn1', 'layer3.8.relu', 'layer3.8.conv2', 'layer3.8.bn2', 'layer3.8.relu_1', 'layer3.8.conv3', 'layer3.8.bn3', 'layer3.8.add', 'layer3.8.relu_2', 'layer3.9.conv1', 'layer3.9.bn1', 'layer3.9.relu', 'layer3.9.conv2', 'layer3.9.bn2', 'layer3.9.relu_1', 'layer3.9.conv3', 'layer3.9.bn3', 'layer3.9.add', 'layer3.9.relu_2', 'layer3.10.conv1', 'layer3.10.bn1', 'layer3.10.relu', 'layer3.10.conv2', 'layer3.10.bn2', 'layer3.10.relu_1', 'layer3.10.conv3', 'layer3.10.bn3', 'layer3.10.add', 'layer3.10.relu_2', 'layer3.11.conv1', 'layer3.11.bn1', 'layer3.11.relu', 'layer3.11.conv2', 'layer3.11.bn2', 'layer3.11.relu_1', 'layer3.11.conv3', 'layer3.11.bn3', 'layer3.11.add', 'layer3.11.relu_2', 'layer3.12.conv1', 'layer3.12.bn1', 'layer3.12.relu', 'layer3.12.conv2', 'layer3.12.bn2', 'layer3.12.relu_1', 'layer3.12.conv3', 'layer3.12.bn3', 'layer3.12.add', 'layer3.12.relu_2', 'layer3.13.conv1', 'layer3.13.bn1', 'layer3.13.relu', 'layer3.13.conv2', 'layer3.13.bn2', 'layer3.13.relu_1', 'layer3.13.conv3', 'layer3.13.bn3', 'layer3.13.add', 'layer3.13.relu_2', 'layer3.14.conv1', 'layer3.14.bn1', 'layer3.14.relu', 'layer3.14.conv2', 'layer3.14.bn2', 'layer3.14.relu_1', 'layer3.14.conv3', 'layer3.14.bn3', 'layer3.14.add', 'layer3.14.relu_2', 'layer3.15.conv1', 'layer3.15.bn1', 'layer3.15.relu', 'layer3.15.conv2', 'layer3.15.bn2', 'layer3.15.relu_1', 'layer3.15.conv3', 'layer3.15.bn3', 'layer3.15.add', 'layer3.15.relu_2', 'layer3.16.conv1', 'layer3.16.bn1', 'layer3.16.relu', 'layer3.16.conv2', 'layer3.16.bn2', 'layer3.16.relu_1', 'layer3.16.conv3', 'layer3.16.bn3', 'layer3.16.add', 'layer3.16.relu_2', 'layer3.17.conv1', 'layer3.17.bn1', 'layer3.17.relu', 'layer3.17.conv2', 'layer3.17.bn2', 'layer3.17.relu_1', 'layer3.17.conv3', 'layer3.17.bn3', 'layer3.17.add', 'layer3.17.relu_2', 'layer3.18.conv1', 'layer3.18.bn1', 'layer3.18.relu', 'layer3.18.conv2', 'layer3.18.bn2', 'layer3.18.relu_1', 'layer3.18.conv3', 'layer3.18.bn3', 'layer3.18.add', 'layer3.18.relu_2', 'layer3.19.conv1', 'layer3.19.bn1', 'layer3.19.relu', 'layer3.19.conv2', 'layer3.19.bn2', 'layer3.19.relu_1', 'layer3.19.conv3', 'layer3.19.bn3', 'layer3.19.add', 'layer3.19.relu_2', 'layer3.20.conv1', 'layer3.20.bn1', 'layer3.20.relu', 'layer3.20.conv2', 'layer3.20.bn2', 'layer3.20.relu_1', 'layer3.20.conv3', 'layer3.20.bn3', 'layer3.20.add', 'layer3.20.relu_2', 'layer3.21.conv1', 'layer3.21.bn1', 'layer3.21.relu', 'layer3.21.conv2', 'layer3.21.bn2', 'layer3.21.relu_1', 'layer3.21.conv3', 'layer3.21.bn3', 'layer3.21.add', 'layer3.21.relu_2', 'layer3.22.conv1', 'layer3.22.bn1', 'layer3.22.relu', 'layer3.22.conv2', 'layer3.22.bn2', 'layer3.22.relu_1', 'layer3.22.conv3', 'layer3.22.bn3', 'layer3.22.add', 'layer3.22.relu_2', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.relu_1', 'layer4.0.conv3', 'layer4.0.bn3', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.0.add', 'layer4.0.relu_2', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.relu_1', 'layer4.1.conv3', 'layer4.1.bn3', 'layer4.1.add', 'layer4.1.relu_2', 'layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.relu', 'layer4.2.conv2', 'layer4.2.bn2', 'layer4.2.relu_1', 'layer4.2.conv3', 'layer4.2.bn3', 'layer4.2.add', 'layer4.2.relu_2', 'avgpool', 'flatten', 'fc'],
        "vgg19":  ['features.4', 'features.9','features.18','features.27','features.36'],
        "resnet18":  ['x', 'conv1', 'bn1', 'relu', 'maxpool', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.add', 'layer1.0.relu_1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.add', 'layer1.1.relu_1', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.0.add', 'layer2.0.relu_1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.add', 'layer2.1.relu_1', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.0.add', 'layer3.0.relu_1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.add', 'layer3.1.relu_1', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.0.add', 'layer4.0.relu_1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.add', 'layer4.1.relu_1', 'avgpool', 'flatten', 'fc'],
    }
    return pool_layers_dict[model_name]



def get_model_name(job_id: int):
    # id_to_model_name = {
    #     0: "alexnet",
    #     1: "resnet50",
    #     2: "resnet101",
    #     3: "vgg16",
    #     4: "vgg19",
    #     5: "inception_v3",
    #     6: "vit_b_16",
    #     7: "vit_b_32",
    #     8: "efficientnet_v2_s",
    #     9: "resnet18",
    # }

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

    # id_to_model_name = {
    #     0: "alexnet",
    #     1: "inception_v3",
    # }

    # id_to_model_name = {
    #     0: "vit_b_16",
    #     1: "vit_b_32",
    #     2: "efficientnet_v2_s",
    #     3: "resnet18",
    # }

    return id_to_model_name[job_id]


def save_h5(feature_path, model_name, task_name, layer_name, dst_path: str):

    #resnet_no = 1

    with open(
        #os.path.join(feature_path, f"{model_name}_{task_name}_{resnet_no}.pkl"), "rb"
        os.path.join(feature_path, f"{model_name}_{task_name}.pkl"), "rb"
    ) as fin:
        features = pickle.load(fin)

    data = features[layer_name]
    trials = data.shape[0]
    data = data.reshape(trials, -1)

    print(
        f"Model: {model_name}, Task: {task_name}, Layer: {layer_name}, Shape: {data.shape}"
    )

    # Save the NumPy array as a .h5 file
    with h5py.File(os.path.join(dst_path, f"{model_name}_{task_name}.h5"), "w") as hf:
        hf.create_dataset("features", data=data)


def save_task(feature_path, model_name, task, dst_path):
    #layer_name = get_it_layer(model_name)
    
    pool_layers = get_pool_layers(model_name)
    save_h5(feature_path, model_name, task, pool_layers, dst_path)


def main(args):

    multiprocess: bool = False

    #tasks = ["imagenet", "lamem", "lamem_shuffle"]
    tasks = [ "lamem"]
    # tasks = ["lamem_pretrain_freeze", "lamem_pretrain_no_freeze"]
    # tasks = ["lamem_random_pretrain_no_freeze"]
    # tasks = ["lamem_shuffle_pretrain_freeze"]
    
    

    model_names = args.model_names
    feature_path = args.src_path
    node_id = args.node_id

    dst_path = args.dst_path
    os.makedirs(dst_path, exist_ok=True)

    if multiprocess:

        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = []
            for model_name in model_names:
                for task in tasks:
                    futures.append(
                        executor.submit(
                            save_task, feature_path, model_name, task, dst_path
                        )
                    )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing"
            ):
                future.result()  # this will raise any exceptions encountered during execution

    else:
        #model_name = get_model_name(node_id - 1)
        for model_name in model_names:
            for task in tasks:
                save_task(feature_path, model_name, task, dst_path)

        # for model_name in model_names:
        #     for task in tasks:
        #         layer_name = get_it_layer(model_name)
        #         save_h5(feature_path, model_name, task, layer_name, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pkl files to h5 files")

    # argument for model_name,
    # argument for model_name,
    parser.add_argument(
        "--model_names", type=str, nargs="+", required=False, help="Model name"
    )
    # argument for dst_path
    parser.add_argument("--dst_path", type=str, required=True, help="dst path name")
    parser.add_argument("--src_path", type=str, required=True, help="src path name")
    parser.add_argument("--node_id", type=int, required=False, help="node_id")

    args = parser.parse_args()

    main(args)


# python load_feature_pkl_files_and_convert_to_h5.py --model_names resnet50 resnet101 alexnet vgg16 vgg19 inception_v3 --dst_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features_h5 --src_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features
