import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
import os

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
sys.path.append(parent_dir)

from datasets.ImageNet import ImageNet
from datasets.ImageNet import parse_devkit_archive


def main(args):
    # Create ImageNet dataset
    # parse_devkit_archive(args.root)

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageNet(
        root=args.root,
        split="train",
        temp_extract=args.temp_extract,
        transform=transform,
    )

    print(f"Number of images: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    batch = next(iter(dataloader))

    print(batch)

    # Process the data
    for batch in tqdm(dataloader, desc="Processing ImageNet"):
        # Your processing logic here
        pass

    dataset = ImageNet(
        root=args.root,
        split="val",
        temp_extract=args.temp_extract,
        transform=transform,
    )

    print(f"Number of images: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    batch = next(iter(dataloader))

    print(batch)

    # Process the data
    for batch in tqdm(dataloader, desc="Processing ImageNet"):
        # Your processing logic here
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ImageNet dataset")
    parser.add_argument(
        "--root", type=str, required=True, help="Path to ImageNet root directory"
    )
    parser.add_argument(
        "--temp_extract",
        action="store_true",
        help="Extract files to temporary directory",
    )
    args = parser.parse_args()

    main(args)
