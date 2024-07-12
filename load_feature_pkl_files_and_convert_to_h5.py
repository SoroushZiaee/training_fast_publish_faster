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

    resnet_no = 1

    with open(
        os.path.join(feature_path, f"{model_name}_{task_name}_{resnet_no}.pkl"), "rb"
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
    layer_name = get_it_layer(model_name)
    save_h5(feature_path, model_name, task, layer_name, dst_path)


def main(args):

    multiprocess: bool = False

    # tasks = ["imagenet", "lamem", "lamem_shuffle"]
    # tasks = ["lamem_pretrain_freeze", "lamem_pretrain_no_freeze"]
    tasks = ["lamem_random_pretrain_no_freeze"]

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
        model_name = get_model_name(node_id - 1)
        for task in tasks:
            layer_name = get_it_layer(model_name)
            save_h5(feature_path, model_name, task, layer_name, dst_path)

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
