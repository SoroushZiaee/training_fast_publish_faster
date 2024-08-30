import argparse
import os
import pickle
from typing import List
import numpy as np
from joblib import Parallel, delayed

from tqdm import tqdm
from memory_profiler import profile
from itertools import combinations


def get_layer_name(model_name: str) -> List[str]:
    layer_dict = {
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


def load_features(feature_path, model_name, task_name, model_num: int = 1):
    with open(
        os.path.join(feature_path, f"{model_name}_{task_name}_{model_num}.pkl"), "rb"
    ) as fin:
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
    print(f"{cka_value = }")

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

    model_ids = [i for i in range(1, 13)]
    comb_model_ids = list(combinations(model_ids, 2))
    comb_model_ids = [comb_model_ids[args.job_id - 1]]  # since job_id start from 1

    for id1, id2 in comb_model_ids:

        print(f"{id1 = }, {id2 = }")

        # if os.path.exists(
        #     os.path.join(
        #         dst_path, f"{model_name}_{task1}_{task1}_({id1},{id2})_cka.pkl"
        #     )
        # ):
        #     print("File Exists")
        #     continue

        features_1 = load_features(args.src_path, model_name, task1, id1)
        features_2 = load_features(args.src_path, model_name, task1, id2)

        print(f"{features_1.keys()=}, {features_2.keys()=}")

        n_jobs = len(layer_names)
        print(f"{n_jobs=}")

        results = []
        for layer in tqdm(layer_names):
            results.append((layer, get_cka_score(layer, features_1, features_2)))

        results = dict(
            results
        )  # convert to dictionary, key = layer_name, value = cka_value

        print(f"{results =}")

        with open(
            os.path.join(
                dst_path, f"{model_name}_{task1}_{task1}_({id1},{id2})_cka.pkl"
            ),
            "wb",
        ) as fout:
            pickle.dump(results, fout)

        del features_1, features_2

        clear_memory()
        print("Memory Cleared")

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--dst_path", type=str, required=True)
    parser.add_argument("--job_id", type=int, required=False)  # for slurm

    args = parser.parse_args()

    main(args)

# python calculate_cka_for_resnet50.py --model_name resnet50 --src_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/model_features --dst_path /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/cka_results
