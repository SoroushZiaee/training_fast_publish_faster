import os

if __name__ == "__main__":
    folder_names_rename_version = {
        "1ee0e33e-8901-4a89-a98e-309060eea088": "resnet50-4",
        "2f039a8f-1fe4-4f9a-954e-0fe4cef5d895": "resnet50-5",
        "50fc0d4d-b393-4066-bfd8-e8974a6ae17e": "resnet50-6",
        "54e3f170-308c-46d5-b2c8-ededc2ce301c": "resnet50-7",
        "2780f471-a987-4e5b-abef-bbfe5e79e51b": "resnet50-8",
        "a64a646a-d45d-4715-a8ff-d84edb9c16dd": "resnet50-9",
        "bb7f2c0e-8bb7-4eac-8da5-8c19dd77d322": "resnet50-10",
        "d8d4a882-db53-4f47-a1b8-841380ae5baa": "resnet50-11",
        "f80cac1a-11c6-4983-96cd-910d9ffb2344": "resnet50-12",
    }

    logs_path = (
        "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/logs"
    )
    weights_path = "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/weights"

    for real_name, rename_to in folder_names_rename_version.items():
        real_path = os.path.join(logs_path, real_name)
        rename_path = os.path.join(logs_path, rename_to)
        os.rename(real_path, rename_path)

        real_path = os.path.join(weights_path, real_name)
        rename_path = os.path.join(weights_path, rename_to)
        os.rename(real_path, rename_path)
