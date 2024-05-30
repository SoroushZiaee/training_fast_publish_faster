import tarfile
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse


def extract_nested_tar(nested_tar_info, tar_file_path, extract_dir):
    with tarfile.open(tar_file_path, "r") as tar_file:
        nested_tar_file_obj = tar_file.extractfile(nested_tar_info)
        if nested_tar_file_obj is None:
            return

        folder_name = os.path.join(
            extract_dir, os.path.splitext(nested_tar_info.name)[0]
        )
        os.makedirs(folder_name, exist_ok=True)

        nested_tar_path = os.path.join(folder_name, nested_tar_info.name)
        with open(nested_tar_path, "wb") as f:
            f.write(nested_tar_file_obj.read())

        with tarfile.open(nested_tar_path, "r") as nested_tar_file:
            nested_tar_file.extractall(folder_name)

        os.remove(nested_tar_path)


def extract_tar_files(tar_file_path, extract_dir, n_jobs=-1):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    with tarfile.open(tar_file_path, "r") as tar_file:
        members = [
            member
            for member in tar_file.getmembers()
            if member.isfile()
            and member.name.endswith(".tar")
            and not os.path.exists(
                os.path.join(extract_dir, os.path.splitext(member.name)[0])
            )
        ]
        print(len(members))

    Parallel(n_jobs=n_jobs)(
        delayed(extract_nested_tar)(member, tar_file_path, extract_dir)
        for member in tqdm(members)
    )


def main(tar_file_path: str, extract_dir: str):
    # Number of parallel jobs (use -1 to use all CPUs, or specify a number)
    n_jobs = 30

    # Extract the tar files
    extract_tar_files(tar_file_path, extract_dir, n_jobs)


if __name__ == "__main__":
    # write a code to take argument from user for tar_file_path and extract_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("tar_file_path", type=str, help="Path to the tar file")
    parser.add_argument(
        "extract_dir", type=str, help="Directory to extract the contents to"
    )
    args = parser.parse_args()
    main(args.tar_file_path, args.extract_dir)


# Write a sample running command for this script
# python extract_script.py /home/soroush1/projects/def-kohitij/soroush1/imagenet/ILSVRC2012_img_train.tar /home/soroush1/nearline/def-kohitij/soroush1/train_folder
