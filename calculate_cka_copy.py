#!/usr/bin/env python

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
        
        "resnet101": ['maxpool', 'layer1.1.add', 'layer2.0.add', 'layer2.3.add', 'layer3.1.add', 'layer3.4.add', 'layer3.7.add', 'layer3.10.add', 'layer3.13.add', 'layer3.16.add'],

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
        
        "resnet50": ['maxpool', 'layer1.0.add', 'layer1.2.add', 'layer2.0.add', 'layer2.2.add', 'layer3.0.downsample.0', 'layer3.1.add', 'layer3.3.add', 'layer3.5.add', 'layer4.0.add'],
        "resnet18": ['maxpool', 'layer1.0.add', 'layer1.1.add', 'layer2.0.add', 'layer2.1.add', 'layer3.0.add', 'layer3.1.add', 'layer4.0.add', 'layer4.1.add', 'avgpool'],
        
        "efficientnet_v2_s": ['features.1.0.add', 'features.2.2.add', 'features.3.2.add', 'features.4.2.add', 'features.4.5.add', 'features.5.3.add', 'features.5.6.add', 'features.6.1.add', 'features.6.4.add', 'features.6.7.add'],
    
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
    with open(os.path.join(feature_path, f"{model_name}_{task_name}_1.pkl"), "rb") as fin:
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
    task1, task2 = "lamem_shuffle", "lamem"
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
