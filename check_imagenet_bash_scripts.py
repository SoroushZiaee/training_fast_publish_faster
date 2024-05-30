import os
import argparse


def main(folder: str):
    print(f"{len(os.listdir(folder)) = }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check the bash scripts for the ImageNet dataset"
    )
    parser.add_argument("--folder", type=str, help="The bash script to check")

    # Take the arguements
    args = parser.parse_args()

    main(args.folder)
