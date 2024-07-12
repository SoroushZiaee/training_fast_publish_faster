import argparse
import os
import pickle
from typing import List
import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm
from memory_profiler import profile


def get_layer_name(model_name: str) -> List[str]:
    layer_dict = {
        "resnet101": [
            "x",
            "layer2.0.bn1",
            "layer2.3.bn3",
            "layer3.3.bn2",
            "layer3.7.relu",
            "layer3.11.conv1",
            "layer3.14.relu_2",
            "layer3.18.bn3",
            "layer3.22.relu_1",
            "avgpool",
            "layer3.2.bn1",
        ],
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
        "vgg19": [
            "x",
            "features.4",
            "features.9",
            "features.14",
            "features.19",
            "features.24",
            "features.29",
            "features.34",
            "classifier.0",
            "classifier.5",
            "features.36",
        ],
        "inception_v3": [
            "x",
            "Mixed_5b.branch5x5_2.conv",
            "Mixed_5c.branch_pool.bn",
            "Mixed_6a.max_pool2d",
            "Mixed_6c.branch7x7_1.conv",
            "Mixed_6d.branch7x7_3.conv",
            "Mixed_6e.branch7x7dbl_1.relu",
            "Mixed_7a.max_pool2d",
            "Mixed_7b.branch_pool.conv",
            "avgpool",
            "Mixed_7a.branch3x3_1.bn",
        ],
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
        "resnet50": [
            "x",
            "layer1.1.relu",
            "layer2.0.bn1",
            "layer2.1.relu_2",
            "layer2.3.add",
            "layer3.1.conv3",
            "layer3.3.relu_1",
            "layer3.5.relu_1",
            "layer4.1.relu",
            "fc",
            "layer3.2.bn1",
        ],
        "resnet18": [
            "x",
            "layer1.0.relu",
            "layer1.1.conv2",
            "layer2.0.bn2",
            "layer2.1.relu",
            "layer3.0.conv2",
            "layer3.1.relu",
            "layer4.0.relu",
            "layer4.1.bn1",
            "fc",
        ],
        "efficientnet_v2_s": [
            "x",
            "features.3.0.block.1",
            "features.4.2.block.1",
            "features.5.0.block.1",
            "features.5.4.block.3",
            "features.5.8.block.2",
            "features.6.3.stochastic_depth",
            "features.6.7.stochastic_depth",
            "features.6.11.stochastic_depth",
            "classifier.1",
        ],
    }

    return layer_dict[model_name]


def unbiased_HSIC(K, L):
    """Computes an unbiased estimator of HISC. This is equation (2) from the paper"""

    # create the unit **vector** filled with ones
    n = K.shape[0]
    ones = np.ones(shape=(n))

    # fill the diagonal entries with zeros
    np.fill_diagonal(K, val=0)  # this is now K_tilde
    np.fill_diagonal(L, val=0)  # this is now L_tilde

    # first part in the square brackets
    trace = np.trace(np.dot(K, L))

    # middle part in the square brackets
    nominator1 = np.dot(np.dot(ones.T, K), ones)
    nominator2 = np.dot(np.dot(ones.T, L), ones)
    denominator = (n - 1) * (n - 2)
    middle = np.dot(nominator1, nominator2) / denominator

    # third part in the square brackets
    multiplier1 = 2 / (n - 2)
    multiplier2 = np.dot(np.dot(ones.T, K), np.dot(L, ones))
    last = multiplier1 * multiplier2

    # complete equation
    unbiased_hsic = 1 / (n * (n - 3)) * (trace + middle - last)

    return unbiased_hsic


def CKA(X, Y):
    """Computes the CKA of two matrices. This is equation (1) from the paper"""

    nominator = unbiased_HSIC(np.dot(X, X.T), np.dot(Y, Y.T))
    denominator1 = unbiased_HSIC(np.dot(X, X.T), np.dot(X, X.T))
    denominator2 = unbiased_HSIC(np.dot(Y, Y.T), np.dot(Y, Y.T))

    cka = nominator / np.sqrt(denominator1 * denominator2)

    return cka


def load_features(feature_path, model_name, task_name):
    with open(os.path.join(feature_path, f"{model_name}_{task_name}.pkl"), "rb") as fin:
        features = pickle.load(fin)

    return features


def get_layer_features(features, layer_name):

    feature_shape = features[layer_name].shape
    print(f"{feature_shape = }")

    return features[layer_name].reshape(feature_shape[0], -1)


def clear_memory():
    # write a code to flush memory, and free up the memory
    import gc

    gc.collect()


def get_cka_score(
    layer_name: str, features_1: np.ndarray, features_2: np.ndarray
) -> float:
    layer_features_1 = get_layer_features(features_1, layer_name)
    layer_features_2 = get_layer_features(features_2, layer_name)

    print(f"{layer_features_1.shape = }")
    print(f"{layer_features_2.shape = }")

    cka_value = CKA(layer_features_1, layer_features_2)

    print(f"{layer_name=}, {cka_value=}")

    del layer_features_1, layer_features_2
    print("Memory Cleared")
    clear_memory()

    return (layer_name, cka_value)


@profile
def main(args: argparse.Namespace = None):
    task1, task2 = "imagenet", "lamem"
    model_name = args.model_name
    multiprocess = False

    layer_names = get_layer_name(model_name)

    dst_path = args.dst_path
    os.makedirs(dst_path, exist_ok=True)

    print(
        f"{layer_names=}, {len(layer_names)=}, {model_name=}, {task1=}, {task2=}, \n{args.src_path=}"
    )

    features_1 = load_features(args.src_path, model_name, task1)
    features_2 = load_features(args.src_path, model_name, task2)

    print(f"{features_1.keys()=}, {features_2.keys()=}")

    n_jobs = len(layer_names)
    print(f"{n_jobs=}")

    if multiprocess:
        results = Parallel(n_jobs=n_jobs)(
            delayed(get_cka_score)(layer, features_1, features_2)
            for layer in layer_names
        )

    results = []
    for layer in tqdm(layer_names):
        results.append((layer, get_cka_score(layer, features_1, features_2)))

    results = dict(
        results
    )  # convert to dictionary, key = layer_name, value = cka_value

    with open(
        os.path.join(dst_path, f"{model_name}_{task1}_{task2}_cka.pkl"), "wb"
    ) as fout:
        pickle.dump(results, fout)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--dst_path", type=str, required=True)

    args = parser.parse_args()

    main(args)

# python calculate_cka.py --model_name resnet101 --src_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features --dst_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/cka_results
