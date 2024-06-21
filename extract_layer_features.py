from math import comb
from typing import List, Optional, Optional
import torch
import torch as ch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import models
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from tqdm import tqdm

import numpy as np

import os
import PIL.Image
import argparse
from memory_profiler import profile


class MuriDataset(Dataset):
    def __init__(self, root: str, transforms=None):

        self.root = root
        self.transforms = transforms
        img_path_list = os.listdir(root)
        img_path_list.sort()

        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.img_path_list[idx])
        # print(f"{img_name = }")
        image = PIL.Image.open(img_name).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image


class BlurPoolConv2d(torch.nn.Module):

    # Purpose: This class creates a convolutional layer that first applies a blurring filter to the input before performing the convolution operation.
    # Condition: The function apply_blurpool iterates over all layers of the model and replaces convolution layers (ch.nn.Conv2d) with BlurPoolConv2d if they have a stride greater than 1 and at least 16 input channels.
    # Preventing Aliasing: Blurring the output of convolution layers (especially those with strides greater than 1) helps to reduce aliasing effects. Aliasing occurs when high-frequency signals are sampled too sparsely, leading to incorrect representations.
    # Smooth Transitions: Applying a blur before downsampling ensures that transitions between pixels are smooth, preserving important information in the feature maps.
    # Stabilizing Training: Blurring can help stabilize training by reducing high-frequency noise, making the model less sensitive to small changes in the input data.
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


class RegressionModel(ch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # model = self.modified_inplace_problem(model)
        out_features = self.get_last_layer_features(model)
        self.model = model
        self.regression_layer = ch.nn.Sequential(
            ch.nn.Linear(
                out_features, 1
            ),  # Adjust this if your final layer is different
            ch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.regression_layer(x)
        x = x.squeeze()
        return x

    @staticmethod
    def get_last_layer_features(model: ch.nn.Module) -> int:
        """
        Get the number of input features for the last linear layer of the model.

        Args:
            model (nn.Module): The neural network model.

        Returns:
            int: The number of input features for the last linear layer.
        """
        for layer in reversed(list(model.children())):
            if isinstance(layer, ch.nn.Sequential):
                for sub_layer in reversed(list(layer.children())):
                    if isinstance(sub_layer, ch.nn.Linear):
                        return sub_layer.out_features
            elif isinstance(layer, ch.nn.Linear):
                return layer.out_features
        raise ValueError("No linear layer found in the model")


def apply_blurpool(mod: torch.nn.Module):
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (
            np.max(child.stride) > 1 and child.in_channels >= 16
        ):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def apply_regression(model):
    return RegressionModel(model)


def get_checkpoint_path(model_name: str, task: str):
    path_dict = {
        "resnet101": {
            "imagenet": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/weights/clf/resnet101-1/checkpoint_epoch_90_0.78.pth",
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/resnet101/epoch=89-val_loss=0.01-training_loss=0.01.ckpt",
            "lamem_shuffle": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem_random/alexnet/epoch=54-val_loss=0.02-training_loss=0.02.ckpt",
            "combine": "alexnet_combine.pth",
        },
        "resnet50": {
            "imagenet": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/weights/clf/resnet50_weights/resnet50-1/checkpoint_epoch_90_0.77.pth",
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/resnet50/epoch=74-val_loss=0.01-training_loss=0.01.ckpt",
            "lamem_shuffle": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem_random/resnet50/epoch=89-val_loss=0.02-training_loss=0.02.ckpt",
            "combine": "alexnet_combine.pth",
        },
        "resnet18": {
            "imagenet": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/weights/clf/resnet18_weights/resnet18-0/checkpoint_epoch_90_0.70.pth",
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/resnet18/epoch=69-val_loss=0.01-training_loss=0.00.ckpt",
            "combine": "alexnet_combine.pth",
        },
        "vgg16": {
            "imagenet": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/weights/clf/vgg16_weights/vgg16-0/checkpoint_epoch_90_0.63.pth",
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/vgg16/epoch=9-val_loss=0.01-training_loss=0.01.ckpt",
            "combine": "alexnet_combine.pth",
        },
        "vgg19": {
            "imagenet": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/weights/clf/vgg19_weights/vgg19-0/checkpoint_epoch_90_0.63.pth",
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/vgg19/epoch=9-val_loss=0.01-training_loss=0.01.ckpt",
            "combine": "alexnet_combine.pth",
        },
        "inception_v3": {
            "imagenet": None,  # fix this
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/inception/epoch=9-val_loss=0.01-training_loss=0.01.ckpt",
            "combine": "alexnet_combine.pth",
        },
        "alexnet": {
            "imagenet": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/weights/clf/alexnet_weights/alexnet-1/checkpoint_epoch_90_0.52.pth",
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/alexnet/epoch=89-val_loss=0.01-training_loss=0.01.ckpt",  # fix this
            "combine": "alexnet_combine.pth",
        },
        "vit_b_16": {
            "imagenet": None,  # fix this
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/vit_b_16/epoch=9-val_loss=0.01-training_loss=0.01.ckpt",
            "combine": "alexnet_combine.pth",
        },
        "vit_b_32": {
            "imagenet": None,  # fix this
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/vit_b_32/epoch=9-val_loss=0.01-training_loss=0.01.ckpt",
            "combine": "alexnet_combine.pth",
        },
        "efficientnet_v2_s": {
            "imagenet": None,  # fix this
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/efficient_v2/epoch=9-val_loss=0.01-training_loss=0.01.ckpt",
            "combine": "alexnet_combine.pth",
        },
    }

    return path_dict[model_name][task]


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
            "x",
            "cat",
            "encoder.layers.encoder_layer_1.dim",
            "encoder.layers.encoder_layer_2.dropout",
            "encoder.layers.encoder_layer_4.getattr",
            "encoder.layers.encoder_layer_5.ln_1",
            "encoder.layers.encoder_layer_7.self_attention",
            "encoder.layers.encoder_layer_8.add_1",
            "encoder.layers.encoder_layer_10.getitem_1",
            "heads.head",
            "encoder.layers.encoder_layer_3.mlp",
        ],
        "vit_b_32": [
            "x",
            "cat",
            "encoder.layers.encoder_layer_1.dim",
            "encoder.layers.encoder_layer_2.dropout",
            "encoder.layers.encoder_layer_4.getattr",
            "encoder.layers.encoder_layer_5.ln_1",
            "encoder.layers.encoder_layer_7.self_attention",
            "encoder.layers.encoder_layer_8.add_1",
            "encoder.layers.encoder_layer_10.getitem_1",
            "heads.head",
            "encoder.layers.encoder_layer_6.mlp",
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


def get_prefix(task: str = "imagenet", model_name: str = "alexnet"):
    if task == "imagenet":
        return "module."

    elif task == "lamem":
        return "model.model."

    else:
        raise NotImplementedError


def remove_prefix(state_dict, prefix):
    """
    Remove a prefix from the state_dict keys.

    Args:
    state_dict (dict): State dictionary from which the prefix will be removed.
    prefix (str): Prefix to be removed.

    Returns:
    dict: State dictionary with prefix removed from keys.
    """
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def get_dataset(root: str, input_size: int = 256):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return MuriDataset(root=root, transforms=imagenet_transform)


def match_and_load_weights(checkpoint_state_dict, model, prefix="module."):
    """
    Match weights from checkpoint_state_dict with model's state_dict and load them into the model.

    Args:
    checkpoint_state_dict (dict): State dictionary from checkpoint.
    model (torch.nn.Module): The model instance.
    prefix (str): Prefix to be removed from checkpoint keys.

    Returns:
    None
    """
    # Remove the prefix from checkpoint state dict keys
    cleaned_checkpoint_state_dict = remove_prefix(checkpoint_state_dict, prefix)

    model_state_dict = model.state_dict()
    matched_weights = {}

    # Iterate over the cleaned checkpoint state dict
    for ckpt_key, ckpt_weight in cleaned_checkpoint_state_dict.items():
        if ckpt_key in model_state_dict:
            # If the layer name matches, add to the matched_weights dict
            matched_weights[ckpt_key] = ckpt_weight
        else:
            print(
                f"Layer {ckpt_key} from checkpoint not found in the model state dict."
            )

    return matched_weights


def get_model(
    model_name,
    checkpoint_path: str = None,
    layer_name: Optional[List[str]] = None,
    use_blurpool: bool = True,
    task: str = "imagenet",
    device="cpu",
):
    """
    Create a model from torchvision.models and load weights from checkpoint if provided.

    Args:
    model_name (str): Name of the model to be created.
    checkpoint_path (str): Path to the checkpoint file.
    layer_name (str): Name of the layer to extract features from.
    use_blurpool (bool): Whether to use BlurPoolConv2d for convolution layers with stride > 1.
    task (str): Whether to be imagenet, lamem, or combine


    Returns:
    torch.nn.Module: The model instance.
    """

    def correct_checkpoint(checkpoint):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        return checkpoint

    if isinstance(layer_name, str):
        layer_name = [layer_name]

    model = getattr(models, model_name)(weights=None)
    print(f"{model = }")

    if use_blurpool:
        apply_blurpool(model)

    # TODO: Missing key(s) in state_dict: "model.features.0.weight"
    # if task == "lamem":
    #     model = apply_regression(model)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint = correct_checkpoint(checkpoint)
        # print(f"{checkpoint.keys() = }")
        prefix = get_prefix(task, model_name)
        print(f"{prefix = }")
        # print(f"{checkpoint.keys() = }")
        matched_weights = match_and_load_weights(checkpoint, model, prefix=prefix)
        model.load_state_dict(matched_weights)
        print(f"{checkpoint.keys() = }")

    else:
        print(f"loadding pytorch pre-trained model")
        model = getattr(models, model_name)(weights="DEFAULT")

    if layer_name:
        model = create_feature_extractor(model, layer_name)

    model.to(device)

    return model


def detach_output(output, device: str = "cpu"):
    if isinstance(output, dict):
        for key, value in output.items():
            if device == "cpu":
                output[key] = value.detach().numpy()
            else:
                # print(f"{key = }")
                output[key] = value.detach().cpu().numpy()

    return output


def concantenate_outputs(outputs):
    combined_outputs = {}
    for key in outputs[0].keys():
        combined_outputs[key] = np.concatenate(
            [output[key] for output in outputs], axis=0
        )

    return combined_outputs


def get_inference(model, dataset, device):
    outputs = []
    for i in tqdm(range(len(dataset))):
        x = dataset[i]
        x = x.unsqueeze_(0)
        x = x.to(device)
        output = model(x)
        # print(f"{output = }")
        # print(f"{output.keys() = }")
        # for key, value in output.items():
        #     print(f"{key = }")
        #     print(f"{type(value) = }")

        output = detach_output(output, device)
        outputs.append(output)

        # if i == 2:
        #     break

    outputs = concantenate_outputs(outputs)
    return outputs
    # return np.concatenate(outputs, axis=0)


def clear_memory():
    # write a code to flush memory, and free up the memory
    torch.cuda.empty_cache()
    import gc

    gc.collect()


@profile
def main(args):
    print(f"Extracting features from {args.model} for task {args.task}")

    checkpoint_path = get_checkpoint_path(args.model, args.task)
    layer_names = get_layer_name(args.model)

    print(f"{checkpoint_path = }")
    print(f"layer names:\n{layer_names}")

    root = "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/muri1320"

    input_size = 256
    if args.model == "vit_b_16" or args.model == "vit_b_32":
        print(f"{input_size = }")
        input_size = 224

    ds = get_dataset(root, input_size=input_size)

    print(f"{len(ds) = }")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer_features = {}
    print(f"Extracting features from layers: {layer_names}")
    model = get_model(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        layer_name=layer_names,
        task=args.task,
        device=device,
    )
    model.eval()

    print(f"{model = }")

    sample_input = torch.randn(1, 3, input_size, input_size).to(device)
    print(f"{sample_input.size() = }")

    layer_features = model(sample_input)
    print(f"{type(layer_features) = }")
    print(f"{layer_features.keys() = }")

    layer_features = get_inference(model, ds, device)

    print(f"Saving features to {args.model}_{args.task}")

    # save the pickle file
    import pickle

    with open(f"{args.model}_{args.task}.pkl", "wb") as f:
        pickle.dump(layer_features, f)

    # print(f"{type(layer_features) = }")
    # print(f"{layer_features.keys() = }")
    # print(f"{layer_features['x'].shape = }")
    # print(f"{layer_features['layer3.14.relu_2'].shape = }")

    clear_memory()

    print("Memory Cleared")

    # load the pickle file
    with open(f"{args.model}_{args.task}.pkl", "rb") as f:
        layer_features = pickle.load(f)

    print(f"{layer_features.keys() = }")
    print(f"{layer_features['x'].shape = }")
    # print(f"{layer_features['layer3.14.relu_2'].shape = }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from a layer of a pre-trained model"
    )
    parser.add_argument("--model", type=str, help="Path to the pre-trained model")
    parser.add_argument("--task", type=str, help="Task to perform")

    args = parser.parse_args()
    main(args)


# Example:
# python extract_layer_features.py --model alexnet --task imagenet
# python extract_layer_features.py --model alexnet --task lamem # TODO: re-run to get the correct layer names

# python extract_layer_features.py --model inception_v3 --task lamem
# python extract_layer_features.py --model inception_v3 --task imagenet

# python extract_layer_features.py --model resnet101 --task lamem
# python extract_layer_features.py --model resnet101 --task imagenet

# python extract_layer_features.py --model resnet50 --task lamem
# python extract_layer_features.py --model resnet50 --task imagenet

# python extract_layer_features.py --model vgg19 --task lamem
# python extract_layer_features.py --model vgg19 --task imagenet

# python extract_layer_features.py --model vgg16 --task lamem
# python extract_layer_features.py --model vgg16 --task imagenet

# python extract_layer_features.py --model resnet18 --task lamem
# python extract_layer_features.py --model resnet18 --task imagenet

# python extract_layer_features.py --model efficientnet_v2_s --task lamem
# python extract_layer_features.py --model efficientnet_v2_s --task imagenet

# Error:     output[key] = value.detach().cpu().numpy()
# AttributeError: 'int' object has no attribute 'detach'

# python extract_layer_features.py --model vit_b_16 --task lamem # TODO: can't run this because the layer names are not correct
# python extract_layer_features.py --model vit_b_16 --task imagenet # TODO: can't run this because the layer names are not correct

# python extract_layer_features.py --model vit_b_32 --task lamem # TODO: can't run this because the layer names are not correct
# python extract_layer_features.py --model vit_b_32 --task imagenet # TODO: can't run this because the layer names are not correct
