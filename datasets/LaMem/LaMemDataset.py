import os
from typing import List
import torch
import pandas as pd
from torch.utils.data import Dataset
import PIL.Image
from torchvision import transforms
import numpy as np


class LaMem(Dataset):

    def __init__(self, root: str, splits: List[str], transforms=None, change_labels: bool = False):

        self.mem_frame = pd.concat(
            [
                pd.read_csv(
                    os.path.join(root, "splits", split),
                    delimiter=",",
                )
                for split in splits
            ],
            axis=0,
        ).reset_index(drop=True)
        
        if change_labels:
            print("*" * 100)
            print(f"Changing labels")
            print(f"Before:\n{self.mem_frame.head()}")
            self.mem_frame["memo_score"] = (
                self.mem_frame["memo_score"].sample(frac=1).reset_index(drop=True)
            )
            print(f"After:\n{self.mem_frame.head()}")
            print("*" * 100)

        self.transforms = transforms
        self.images_path = os.path.join(root, "images")

    def __len__(self):
        return len(self.mem_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_path, self.mem_frame["image_name"][idx])
        # image = torch.load(img_name)
        image = PIL.Image.open(img_name).convert("RGB")
        mem_score = self.mem_frame["memo_score"][idx]
        target = float(mem_score)
        target = torch.tensor(target)
        if self.transforms:
            image = self.transforms(image)

        return image, target
