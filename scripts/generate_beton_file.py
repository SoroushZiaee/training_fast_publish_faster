import sys
import os

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
sys.path.append(parent_dir)

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, FloatField

from torch.utils.data import Subset

from datasets.ImageNet.ImageNetDataset import ImageNet
from datasets.LaMem.LaMemDataset import LaMem

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config


Section("cfg", "arguments to give the writer").params(
    dataset=Param(
        And(str, OneOf(["lamem", "imagenet"])),
        "Which dataset to write",
        default="imagenet",
    ),
    split=Param(
        And(str, OneOf(["train", "validation", "test"])),
        "Train or val set",
        required=True,
    ),
    write_path=Param(str, "Where to write the new dataset", required=True),
    write_mode=Param(str, "Mode: raw, smart or jpg", required=False, default="smart"),
    max_resolution=Param(int, "Max image side length", required=True),
    num_workers=Param(int, "Number of workers to use", default=16),
    chunk_size=Param(int, "Chunk size for writing", default=100),
    jpeg_quality=Param(float, "Quality of jpeg images", default=90),
    subset=Param(int, "How many images to use (-1 for all)", default=-1),
    compress_probability=Param(float, "compress probability", default=None),
)


def get_dataset(dataset: str, split: str):

    if dataset == "imagenet":
        root = "/datashare/ImageNet/ILSVRC2012"
        meta_path = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet"
        label_decoder = IntField()
        return (
            ImageNet(
                root=root,
                split=split,
                dst_meta_path=meta_path,
                transform=None,
            ),
            label_decoder,
        )

    elif dataset == "lamem":
        root = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/lamem/lamem_images/lamem"
        splits_list = os.listdir(os.path.join(root, "splits"))
        train_splits = list(sorted(filter(lambda x: "train" in x, splits_list)))
        val_splits = list(sorted(filter(lambda x: "val" in x, splits_list)))
        test_splits = list(sorted(filter(lambda x: "test" in x, splits_list)))
        # Use one Split train_1.csv
        train_splits = [train_splits[0]]
        val_splits = [val_splits[0]]
        test_splits = [test_splits[0]]

        if split == "train":
            splits = train_splits
        elif split == "validation":
            splits = val_splits
        else:
            splits = test_splits

        label_decoder = FloatField()

        return (
            LaMem(
                root=root,
                splits=splits,
                transforms=None,
            ),
            label_decoder,
        )

    return ValueError(f"Wrong {dataset} has been prompted.")


@section("cfg")
@param("dataset")
@param("split")
@param("write_path")
@param("max_resolution")
@param("num_workers")
@param("chunk_size")
@param("subset")
@param("jpeg_quality")
@param("write_mode")
@param("compress_probability")
def main(
    dataset,
    split,
    write_path,
    max_resolution,
    num_workers,
    chunk_size,
    subset,
    jpeg_quality,
    write_mode,
    compress_probability,
):
    my_dataset, label_decoder = get_dataset(dataset=dataset, split=split)
    if subset > 0:
        my_dataset = Subset(my_dataset, range(subset))
        print(f"{len(my_dataset) =}")
    print("here")
    # Pass a type for each data field
    writer = DatasetWriter(
        write_path,
        {
            "image": RGBImageField(
                write_mode=write_mode,
                max_resolution=max_resolution,
                compress_probability=compress_probability,
                jpeg_quality=jpeg_quality,
            ),
            "label": label_decoder,
        },
        num_workers=num_workers,
    )

    # Write dataset
    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size)


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    config.summary()
    main()
