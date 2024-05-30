from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
    NormalizeImage,
    RandomHorizontalFlip,
    ToTorchImage,
)
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields.basics import IntDecoder, FloatDecoder

import torch as ch

import numpy as np
import time

from tqdm import tqdm

print("everything is loaded completely")

train_dataset = "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/lamem_train_256.ffcv"
num_workers = 1
batch_size = 20
distributed = 0
in_memory = True
this_device = "cuda:0"

LAMEM_MEAN = np.load(
    "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/datasets/LaMem/support_files/image_mean.npy"
)

res = 256

decoder = RandomResizedCropRGBImageDecoder((res, res))
image_pipeline = [
    decoder,
    ToTensor(),
    ToDevice(ch.device(this_device), non_blocking=True),
    ToTorchImage(),
    # NormalizeImage(LAMEM_MEAN, IMAGENET_STD, np.float16),
]

label_pipeline = [
    FloatDecoder(),
    ToTensor(),
    Squeeze(),
    ToDevice(ch.device(this_device), non_blocking=True),
]

order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM

loader = Loader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    order=order,
    os_cache=in_memory,
    drop_last=True,
    pipelines={"image": image_pipeline, "label": label_pipeline},
    distributed=distributed,
)


# First epoch includes compilation time
for ims, labs in tqdm(loader):
    pass
start_time = time.time()
for _ in range(100):
    for ims, labs in loader:
        pass
print(f"Shape: {ims.shape} | Time per epoch: {(time.time() - start_time) / 100:.5f}s")
