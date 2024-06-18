import sys
import os
from typing import List

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
sys.path.append(parent_dir)


import torch
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from torchvision import transforms
from tqdm import tqdm
from joblib import Parallel, delayed

from datasets.LaMem.LaMemDataset import LaMem
from datasets.ImageNet.ImageNetDataset import ImageNet


# Define transformations
root = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/lamem/lamem_images/lamem"
resize = 256
desired_image_size = 224

train_transforms_list = transforms.Compose([
    transforms.Resize((resize, resize), PIL.Image.BILINEAR),
    transforms.ToTensor(),
])

# Initialize dataset
obj = LaMem(root, ["train_1.csv", "val_1.csv", "test_1.csv"], transforms=train_transforms_list, change_labels=False)
print(f"{len(obj) = }")

x, y = obj[0]
print(f"{x.shape = }")

print(f"{x.mean() = }")
print(f"{x.max() = }")
print(f"{x.min() = }")

x = transforms.ToPILImage()(x)
x = np.asarray(x)




plt.imshow(x)
plt.savefig("lamem_sample.png")
plt.close()


root = "/datashare/ImageNet/ILSVRC2012"
meta_path = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet"
obj = ImageNet(
                root=root,
                split="train",
                dst_meta_path=meta_path,
                transform=train_transforms_list,
            )

print(f"{len(obj) = }")

x, y = obj[0]
print(f"{x.shape = }")

print(f"{x.mean() = }")
print(f"{x.max() = }")
print(f"{x.min() = }")

x = transforms.ToPILImage()(x)
x = np.asarray(x)

plt.imshow(x)
plt.savefig("image_sample.png")
plt.close()


# # Function to process each image
# def process_image(index):
#     img, _ = obj[index]
#     return img

# # Initialize a zero tensor to accumulate results
# zeros = torch.zeros(size=obj[0][0].shape)

# # Process images in parallel using joblib
# results = Parallel(n_jobs=32)(delayed(process_image)(i) for i in tqdm(range(len(obj)), total=len(obj)))

# # Sum the results
# for i, img in enumerate(results):
#     zeros += img

# zeros /= len(results)
# print(f"{zeros.shape = }")
# print(f"{zeros.min() = }")
# print(f"{zeros.max() = }")

# mean_values = zeros.mean(dim=(1, 2))
# std_values = zeros.std(dim=(1, 2))

# print(f"{mean_values = }")
# print(f"{std_values = }")

# # Concatenate mean and std
# concat_tensor = torch.cat((mean_values, std_values))

# # Save the concatenated tensor to a file
# torch.save(concat_tensor, 'lamem_mean_std_tensor.pt')

# # Verify by loading the tensor back
# loaded_tensor = torch.load('lamem_mean_std_tensor.pt')
# print(f"Loaded tensor: {loaded_tensor}")
