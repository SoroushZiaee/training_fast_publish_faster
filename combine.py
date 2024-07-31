import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist

ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
ch.autograd.set_detect_anomaly(True)

from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
    NormalizeImage,
    RandomHorizontalFlip,
    ToTorchImage,
    Convert
)
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields.basics import IntDecoder, FloatDecoder
from scipy.stats import gmean

class HookExtractor(ch.nn.Module):
    def __init__(self, model, layer_name):
        super(HookExtractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.hook = None
        self.extracted_feature = None

    def hook_fn(self, module, input, output):
        self.extracted_feature = output
    
    def get_layer(self, model, name):
        components = name.split('.')
        for component in components:
            if component.isdigit():
                model = model[int(component)]
            else:
                model = getattr(model, component)
        return model

    def forward(self, x):
        target_layer = self.get_layer(self.model, self.layer_name)
        self.hook = target_layer.register_forward_hook(self.hook_fn)
        _ = self.model(x)
        self.hook.remove()
        return self.extracted_feature
    
class CombineModel(ch.nn.Module):
    def __init__(self, model, layer_name):
        super(CombineModel, self).__init__()
        self.model = model
        self.model = HookExtractor(model, layer_name)
        
        self.avgpool = ch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = ch.nn.Flatten()
        
        # TODO - Avoid hardcoding the number of classes and input channels
        self.clf_head = ch.nn.Linear(512, 1000, bias=False)
        self.reg_head = ch.nn.Sequential(ch.nn.Linear(512, 1, bias=False), ch.nn.Sigmoid())
        
    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        
        clf_out = self.clf_head(x)
        reg_out = self.reg_head(x)
        
        return clf_out, reg_out
        

class BlurPoolConv2d(ch.nn.Module):

    # Purpose: This class creates a convolutional layer that first applies a blurring filter to the input before performing the convolution operation.
    # Condition: The function apply_blurpool iterates over all layers of the model and replaces convolution layers (ch.nn.Conv2d) with BlurPoolConv2d if they have a stride greater than 1 and at least 16 input channels.
    # Preventing Aliasing: Blurring the output of convolution layers (especially those with strides greater than 1) helps to reduce aliasing effects. Aliasing occurs when high-frequency signals are sampled too sparsely, leading to incorrect representations.
    # Smooth Transitions: Applying a blur before downsampling ensures that transitions between pixels are smooth, preserving important information in the feature maps.
    # Stabilizing Training: Blurring can help stabilize training by reducing high-frequency noise, making the model less sensitive to small changes in the input data.
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
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


class MeanScalarMetric(torchmetrics.Metric):
    # Necessity: Ensures that the mean calculation works correctly in a distributed training setup, where metrics need to be aggregated across multiple devices.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("sum", default=ch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=ch.tensor(0), dist_reduce_fx="sum")

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count


class ImageNetTrainer:
    def __init__(self, gpu, config, verbose: bool = True):
        self.gpu = gpu
        self.config = config
        self.uid = str(uuid4())
        self.epoch = 0

        if self.config["distributed"]:
            self.setup_distributed()

        if verbose:
            print("loading dataset...")
        self.train_loader = self.create_train_loader()
        self.lamem_train_loader = self.create_lamem_train_loader()
        self.val_loader = self.create_val_loader()
        self.lamem_val_loader = self.create_lamem_val_loader()

        if verbose:
            print("loading model and optimizers...")
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()

        if self.config["checkpoint"]:
            checkpoint = ch.load(self.config["checkpoint"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            print(f"Model is loaded from the checkpoint.")

        if verbose:
            print("init loggers...")
        self.initialize_logger()
        
        print(f"Initiate testing the model...")
        input_tensor = ch.randn(1, 3, 224, 224)
        input_tensor = input_tensor.to(self.gpu)
        clf_output, reg_output = self.model(input_tensor)
        print(f"{clf_output.size() = }")
        print(f"{reg_output.size() = }")
        

    def setup_distributed(self):
        os.environ["MASTER_ADDR"] = self.config["address"]
        os.environ["MASTER_PORT"] = self.config["port"]

        dist.init_process_group(
            "nccl", rank=self.gpu, world_size=self.config["world_size"]
        )
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def get_lr(self, epoch):
        lr_schedules = {"cyclic": get_cyclic_lr, "step": get_step_lr}
        return lr_schedules[self.config["lr_schedule_type"]](
            epoch,
            lr=self.config["lr"],
            epochs=self.config["epochs"],
            lr_peak_epoch=self.config["lr_peak_epoch"],
        )

    def get_resolution(self, epoch):
        min_res = self.config["min_res"]
        max_res = self.config["max_res"]
        end_ramp = self.config["end_ramp"]
        start_ramp = self.config["start_ramp"]

        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    def create_optimizer(self):
        momentum = self.config["momentum"]
        optimizer = self.config["optimizer"]
        weight_decay = self.config["weight_decay"]
        label_smoothing = self.config["label_smoothing"]

        assert optimizer == "sgd"

        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ("bn" in k)]
        other_params = [v for k, v in all_params if not ("bn" in k)]
        param_groups = [
            {"params": bn_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": weight_decay},
        ]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lamem_loss = ch.nn.MSELoss()
        
    def create_lamem_train_loader(self):
        train_dataset = self.config["lamem_train_dataset"]
        distributed = self.config["distributed"]
        batch_size = self.config["train_batch_size"]
        num_workers = self.config["num_workers"]
        in_memory = self.config["in_memory"]
        
        this_device = f"cuda:{self.gpu}"
        image_pipeline = [
                    self.decoder,
                    RandomHorizontalFlip(),
                    ToTensor(),
                    ToDevice(ch.device(this_device), non_blocking=True),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=13),
                    ToTorchImage(),
                    NormalizeImage(LAMEM_MEAN, LAMEM_STD, np.float16),
                ]
        
        label_pipeline = [
                    FloatDecoder(),
                    ToTensor(),
                    Squeeze(),
                    Convert(ch.float16),
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
        
        return loader
        
        
    def create_train_loader(self):
        train_dataset = self.config["train_dataset"]
        num_workers = self.config["num_workers"]
        batch_size = self.config["train_batch_size"]
        distributed = self.config["distributed"]
        in_memory = self.config["in_memory"]

        this_device = f"cuda:{self.gpu}"
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        label_pipeline = [
            IntDecoder(),
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

        return loader
    
    def create_lamem_val_loader(self):
        val_dataset = self.config["lamem_val_dataset"]
        num_workers = self.config["num_workers"]
        batch_size = self.config["val_batch_size"]
        resolution = self.config["resolution"]
        distributed = self.config["distributed"]

        this_device = f"cuda:{self.gpu}"
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
                    cropper,
                    ToTensor(),
                    ToDevice(ch.device(this_device), non_blocking=True),
                    ToTorchImage(),
                    NormalizeImage(LAMEM_MEAN, LAMEM_STD, np.float16),
                ]

        label_pipeline = [
                    FloatDecoder(),
                    ToTensor(),
                    Squeeze(),
                    Convert(ch.float16),
                    ToDevice(ch.device(this_device), non_blocking=True),
                ]

        loader = Loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )
        return loader

    def create_val_loader(self):
        val_dataset = self.config["val_dataset"]
        num_workers = self.config["num_workers"]
        batch_size = self.config["val_batch_size"]
        resolution = self.config["resolution"]
        distributed = self.config["distributed"]

        this_device = f"cuda:{self.gpu}"
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]

        loader = Loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )
        return loader

    def train(self):
        epochs = self.config["epochs"]
        log_level = self.config["log_level"]

        for epoch in range(self.epoch, epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {"train_loss": f"{str(train_loss)}", "epoch": epoch}
                self.eval_and_log(epoch=epoch, extra_dict=extra_dict)
            
            # if epoch == 0:
            #     break

        self.eval_and_log(extra_dict={"epoch": epoch})
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / "final_weights.pt")

    def eval_and_log(self, epoch: int = 0, extra_dict={}):
        every_n_epochs = self.config["every_n_epochs"]
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(
                dict(
                    {
                        "current_lr": self.optimizer.param_groups[0]["lr"],
                        "top_1": stats["top_1"],
                        "top_5": stats["top_5"],
                        "tr_loss": stats["loss"],
                        "lamem_loss": stats["lamem_loss"],
                        "val_time": val_time,
                    },
                    **extra_dict,
                )
            )

        if self.gpu == 0:
            if epoch % every_n_epochs == 0:
                save_model_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.model_ckpt_path,
                    metric_value=stats["top_1"],
                )
        return stats

    def create_model_and_scaler(self):
        scaler = GradScaler()
        arch = self.config["arch"]
        weights = self.config["weights"]
        use_blurpool = self.config["use_blurpool"]
        checkpoint = self.config["checkpoint"]
        task = "combine"

        model = getattr(models, arch)(weights=weights)

        def apply_blurpool(mod: ch.nn.Module):
            for name, child in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (
                    np.max(child.stride) > 1 and child.in_channels >= 16
                ):
                    setattr(mod, name, BlurPoolConv2d(child))
                else:
                    apply_blurpool(child)
        
        def apply_combine(model):
            return CombineModel(model, "layer4.1.bn2") # resnet 18 - IT layer

        if use_blurpool:
            apply_blurpool(model)
        
        if task == "combine":
            model = apply_combine(model)
            
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if self.config["distributed"]:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    def train_loop(self, epoch):
        model = self.model
        model.train()
        losses = []
        lamem_losses = []
        other_losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = max(len(self.train_loader), len(self.lamem_train_loader))
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(range(iters))
        train_loader_iter = iter(self.train_loader)
        lamem_train_loader_iter = iter(self.lamem_train_loader)

        for ix in tqdm(iterator):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)

            # Main task
            try:
                images, target = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.train_loader)
                images, target = next(train_loader_iter)
                images = images.to(self.gpu)
                target = target.to(self.gpu)
                
                # Check the device of the tensor
                # print(f"{images.device = }")
                # print(f"{target.device = }")
                

            # LaMem task
            try:
                lamem_images, lamem_target = next(lamem_train_loader_iter)
            except StopIteration:
                lamem_train_loader_iter = iter(self.lamem_train_loader)
                lamem_images, lamem_target = next(lamem_train_loader_iter)
                lamem_images = lamem_images.to(self.gpu)
                lamem_target = lamem_target.to(self.gpu)
                # print(f"{lamem_images.device = }")
                # print(f"{lamem_target.device = }")
                
            with ch.autocast(device_type='cuda', dtype=ch.float16):
                # Main task
                clf_out, _ = self.model(images)                
                loss_train = self.loss(clf_out, target)

                # LaMem task
                _, reg_out = self.model(lamem_images)
                reg_out = reg_out.squeeze()
                lamem_loss = self.lamem_loss(reg_out, lamem_target)
                
                # print(f"{loss_train = }")
                # print(f"{lamem_loss = }")
        
                # Combine losses
                total_loss = loss_train + lamem_loss
                
                # print(f"{total_loss = }")

            # Scaling and backward pass
            self.scaler.scale(total_loss).backward()

            # Gradient clipping
            # self.scaler.unscale_(self.optimizer)
            # ch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config["log_level"] > 0:
                losses.append(total_loss.detach().cpu())
                lamem_losses.append(lamem_loss.detach().cpu())
                other_losses.append(loss_train.detach().cpu())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ["ep", "iter", "shape", "lrs"]
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if self.config["log_level"] > 1:
                    names += ["total_loss", "lamem_loss", "other_loss"]
                    values += [f"{total_loss.item():.3f}", f"{lamem_loss.item():.3f}", f"{loss_train.item():.3f}"]

                msg = ", ".join(f"{n}={v}" for n, v in zip(names, values))
                iterator.set_description(msg)
            
            # if ix == 0:
            #     break
        
        # Calculate geometric mean of losses for the entire epoch
        epoch_total_loss = gmean([l.item() for l in losses])
        epoch_lamem_loss = gmean([l.item() for l in lamem_losses])
        epoch_other_loss = gmean([l.item() for l in other_losses])
        
        print(f"{epoch_total_loss = }")
        print(f"{epoch_lamem_loss = }")
        print(f"{epoch_other_loss = }")

        return epoch_total_loss, epoch_lamem_loss, epoch_other_loss

    def val_loop(self):
        model = self.model
        model.eval()
        with ch.no_grad():
            with autocast():
                for ix, (images, target) in tqdm(enumerate(self.val_loader), desc="Val"):
                    clf_output, _ = self.model(images)
                    if self.config["lr_tta"]:
                        clf_output += self.model(ch.flip(images, dims=[3]))[0] # TTA for classification

                    for k in ["top_1", "top_5"]:
                        self.val_meters[k](clf_output, target)

                    loss_val = self.loss(clf_output, target)
                    self.val_meters["loss"](loss_val)
                    
                    # if ix == 0:
                    #     break
                
                for ix, (images, target) in tqdm(enumerate(self.lamem_val_loader), desc="LaMem Val"):
                    _ ,reg_output = self.model(images)
                    
                    print(f"{reg_output = }")
                    print(f"{target = }")
                    
                    loss_val = self.lamem_loss(reg_output, target)
                    print(f"{loss_val = }")
                    
                    print(f"{loss_val.size() = }")
                    self.val_meters["lamem_loss"](loss_val)
                    
                    # if ix == 0:
                    #     break

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    def initialize_logger(self):
        self.val_meters = {
            "top_1": torchmetrics.Accuracy(num_classes=1000).to(self.gpu),
            "top_5": torchmetrics.Accuracy(
                num_classes=1000,
                top_k=5,
            ).to(self.gpu),
            "loss": MeanScalarMetric().to(self.gpu),
            "lamem_loss": MeanScalarMetric().to(self.gpu),
        }

        if self.gpu == 0:
            uuid = str(self.uid)
            self.model_ckpt_path = create_version_dir(
                self.config["model_ckpt_path"], uuid
            )
            folder = (Path(self.config["folder"]) / str(self.uid)).absolute()

            folder.mkdir(parents=True)
            self.log_folder = folder
            self.start_time = time.time()
            print(f"=> Logging in {self.log_folder}")
            params = self.config
            with open(folder / "params.json", "w+") as handle:
                json.dump(params, handle)

    def log(self, content):
        print(f"=> Log: {content}")
        if self.gpu != 0:
            return
        cur_time = time.time()
        with open(self.log_folder / "log", "a+") as fd:
            fd.write(
                json.dumps(
                    {
                        "timestamp": cur_time,
                        "relative_time": cur_time - self.start_time,
                        **content,
                    }
                )
                + "\n"
            )
            fd.flush()

    @classmethod
    def launch(cls, config):
        if config["distributed"]:
            ch.multiprocessing.spawn(
                cls._exec_wrapper,
                nprocs=config["world_size"],
                join=True,
                args=(config,),
            )
        else:
            cls.exec(0, config)

    @classmethod
    def _exec_wrapper(cls, gpu, config):
        cls.exec(gpu, config)

    @classmethod
    def exec(cls, gpu, config):
        trainer = cls(gpu=gpu, config=config)
        if config["eval_only"]:
            trainer.eval_and_log()
        else:
            trainer.train()

        if config["distributed"]:
            trainer.cleanup_distributed()


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256

stats_tensor = ch.load(
    "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/datasets/LaMem/support_files/lamem_mean_std_tensor.pt"
).numpy()
LAMEM_MEAN, LAMEM_STD = stats_tensor[:3] * 255, stats_tensor[3:] * 255


def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr


def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


def get_next_version(base_dir):
    """
    Get the next version number based on the existing directories.

    Parameters:
    base_dir (str): The base directory where versions are stored.

    Returns:
    int: The next version number.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 0

    existing_versions = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    version_numbers = [
        int(d.split("_")[1])
        for d in existing_versions
        if d.startswith("version_") and d.split("_")[1].isdigit()
    ]

    if not version_numbers:
        return 0

    return max(version_numbers) + 1


def create_version_dir(base_dir, uuid):
    """
    Create a new version directory.

    Parameters:
    base_dir (str): The base directory where versions are stored.

    Returns:
    str: The path to the new version directory.
    """
    # next_version = get_next_version(base_dir)
    version_dir = os.path.join(base_dir, uuid)
    os.makedirs(version_dir, exist_ok=True)

    print(f"Created version directory: {version_dir}")
    return version_dir


def save_model_checkpoint(model, optimizer, epoch, version_dir, metric_value):
    """
    Save the model checkpoint with an incremented version number.

    Parameters:
    model (nn.Module): The model to save.
    optimizer (optim.Optimizer): The optimizer state to save.
    epoch (int): The current epoch number.
    version_dir (str): The version directory where checkpoints will be saved.
    """
    checkpoint_path = os.path.join(
        version_dir, f"checkpoint_epoch_{epoch}_{metric_value:.2f}.pth"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    ch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print(f"Model checkpoint saved to: {checkpoint_path}")


def main():
    config = {
        "arch": "resnet18",
        "weights": None,
        "min_res": 160,
        "max_res": 192,
        "end_ramp": 76,
        "start_ramp": 65,
        "train_dataset": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/imagenet_train_256.ffcv",
        "val_dataset": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/imagenet_validation_256.ffcv",
        "lamem_train_dataset": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/lamem_train_256.ffcv",
        "lamem_val_dataset": "/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/lamem_validation_256.ffcv",
        "num_workers": 10,
        "in_memory": 1,
        "step_ratio": 0.1,
        "step_length": 30,
        "lr_schedule_type": "cyclic",
        "lr": 1.7,
        "lr_peak_epoch": 2,
        "folder": "./resnet18_logs",
        "model_ckpt_path": "./resnet18_weights",
        "every_n_epochs": 5,
        "log_level": 1,
        "train_batch_size": 512,
        "val_batch_size": 512,
        "resolution": 224,
        "lr_tta": 1,
        "eval_only": 0,
        "optimizer": "sgd",
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "epochs": 91,
        "label_smoothing": 0.1,
        "distributed": 0,
        "use_blurpool": 1,
        "checkpoint": None,
        "world_size": 1,
        "address": "localhost",
        "port": "12355",
    }

    ImageNetTrainer.launch(config)


if __name__ == "__main__":
    main()
