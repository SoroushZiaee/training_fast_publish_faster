import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist

ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
import torchvision.transforms as T
import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from uuid import uuid4
from typing import List, Optional, Tuple
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
)
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields.basics import IntDecoder

Section("model", "model details").params(
    arch=Param(And(str, OneOf(models.__dir__())), default="resnet18"),
    # weights=Param(str, "is pretrained? (1/0)", default=None),
)

Section("resolution", "resolution scheduling").params(
    min_res=Param(int, "the minimum (starting) resolution", default=160),
    max_res=Param(int, "the maximum (starting) resolution", default=160),
    end_ramp=Param(int, "when to stop interpolating resolution", default=0),
    start_ramp=Param(int, "when to start interpolating resolution", default=0),
    fix_res=Param(int, "using fix resolution or not", default=0),
)

Section("data", "data related stuff").params(
    train_dataset=Param(str, ".dat file to use for training", required=True),
    val_dataset=Param(str, ".dat file to use for validation", required=True),
    num_workers=Param(int, "The number of workers", required=True),
    in_memory=Param(int, "does the dataset fit in memory? (1/0)", required=True),
)

Section("lr", "lr scheduling").params(
    step_ratio=Param(float, "learning rate step ratio", default=0.1),
    step_length=Param(int, "learning rate step length", default=30),
    lr_schedule_type=Param(OneOf(["steplr", "cosineannealinglr"]), default="cyclic"),
    lr_step_size=Param(int, "learning rate step length", default=30),
    lr_gamma=Param(float, "learning rate", default=0.1),
    lr_warmup_epochs=Param(int, "learning rate step length", default=0),
    lr_warmup_method=Param(OneOf(["linear"]), default="linear"),
    lr_warmup_decay=Param(float, "learning rate", default=0.01),
    lr=Param(float, "learning rate", default=0.5),
    lr_min=Param(float, "learning rate", default=0.0),
    lr_peak_epoch=Param(int, "Epoch at which LR peaks", default=2),
)

Section("logging", "how to log stuff").params(
    folder=Param(str, "log location", required=True),
    log_level=Param(int, "0 if only at end 1 otherwise", default=1),
    every_n_epochs=Param(int, "0 if only at end 1 otherwise", default=5),
    model_ckpt_path=Param(str, "model checkpoint path", required=True),
)

Section("validation", "Validation parameters stuff").params(
    batch_size=Param(int, "The batch size for validation", default=512),
    resolution=Param(int, "final resized validation image size", default=224),
    lr_tta=Param(int, "should do lr flipping/avging at test time", default=1),
)

Section("training", "training hyper param stuff").params(
    task=Param(And(str, OneOf(["clf", "reg", "both"])), "training task", default="clf"),
    eval_only=Param(int, "eval only?", default=0),
    batch_size=Param(int, "The batch size", default=512),
    optimizer=Param(And(str, OneOf(["sgd"])), "The optimizer", default="sgd"),
    momentum=Param(float, "SGD momentum", default=0.9),
    weight_decay=Param(float, "weight decay", default=4e-5),
    norm_weight_decay=Param(float, "norm_weight_decay", default=0.0),
    bias_weight_decay=Param(float, "weight decay", default=None),
    transformer_embedding_decay=Param(float, "weight decay", default=None),
    epochs=Param(int, "number of epochs", default=30),
    label_smoothing=Param(float, "label smoothing parameter", default=0.1),
    distributed=Param(int, "is distributed?", default=0),
    use_blurpool=Param(int, "use blurpool?", default=0),
    auto_augment=Param(int, "use auto_augment?", default=0),
    random_erase_prob=Param(float, "random erase prob", default=0.5),
)

Section("dist", "distributed training options").params(
    world_size=Param(int, "number gpus", default=1),
    address=Param(str, "address", default="localhost"),
    port=Param(str, "port", default="12355"),
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


@param("lr.lr")
@param("lr.step_ratio")
@param("lr.step_length")
@param("training.epochs")
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr


@param("lr.lr")
@param("training.epochs")
@param("lr.lr_peak_epoch")
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


def set_weight_decay(
    model: ch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            ch.nn.modules.batchnorm._BatchNorm,
            ch.nn.LayerNorm,
            ch.nn.GroupNorm,
            ch.nn.modules.instancenorm._InstanceNorm,
            ch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = (
                    f"{prefix}.{name}" if prefix != "" and "." in key else name
                )
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups


class BlurPoolConv2d(ch.nn.Module):
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


class ImageNetTrainer:
    @param("training.distributed")
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.create_scheduler(optimizer=self.optimizer)
        self.initialize_logger()

    @param("dist.address")
    @param("dist.port")
    @param("dist.world_size")
    def setup_distributed(self, address, port, world_size):
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param("lr.lr_schedule_type")
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {"cyclic": get_cyclic_lr, "step": get_step_lr}

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param("resolution.min_res")
    @param("resolution.max_res")
    @param("resolution.end_ramp")
    @param("resolution.start_ramp")
    @param("resolution.fix_res")
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp, fix_res):
        if fix_res > 0:
            return fix_res

        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param("lr.lr")
    @param("training.momentum")
    @param("training.optimizer")
    @param("training.weight_decay")
    @param("training.label_smoothing")
    @param("training.norm_weight_decay")
    @param("training.bias_weight_decay")
    @param("training.transformer_embedding_decay")
    def create_optimizer(
        self,
        lr,
        momentum,
        optimizer,
        weight_decay,
        label_smoothing,
        norm_weight_decay,
        bias_weight_decay: float = None,
        transformer_embedding_decay: float = None,
    ):
        assert optimizer == "sgd"

        custom_keys_weight_decay = []
        if bias_weight_decay is not None:
            custom_keys_weight_decay.append(("bias", bias_weight_decay))
        if transformer_embedding_decay is not None:
            for key in [
                "class_token",
                "position_embedding",
                "relative_position_bias_table",
            ]:
                custom_keys_weight_decay.append((key, transformer_embedding_decay))

        param_groups = set_weight_decay(
            self.model,
            weight_decay,
            norm_weight_decay=norm_weight_decay,
            custom_keys_weight_decay=(
                custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None
            ),
        )

        # Only do weight decay on non-batchnorm parameters
        # all_params = list(self.model.named_parameters())
        # bn_params = [v for k, v in all_params if ("bn" in k)]
        # other_params = [v for k, v in all_params if not ("bn" in k)]
        # param_groups = [
        #     {"params": bn_params, "weight_decay": 0.0},
        #     {"params": other_params, "weight_decay": weight_decay},
        # ]

        self.optimizer = ch.optim.SGD(param_groups, lr=lr, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param("lr.lr_schedule_type")
    @param("lr.lr_step_size")
    @param("lr.lr_gamma")
    @param("lr.lr_warmup_epochs")
    @param("lr.lr_warmup_method")
    @param("lr.lr_warmup_decay")
    @param("lr.lr_min")
    @param("training.epochs")
    def create_scheduler(
        self,
        optimizer,
        lr_schedule_type,
        lr_step_size,
        lr_gamma,
        lr_warmup_epochs,
        lr_warmup_method,
        lr_warmup_decay,
        lr_min,
        epochs,
    ):

        if lr_schedule_type == "steplr":
            main_lr_scheduler = ch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_step_size, gamma=lr_gamma
            )

        elif lr_schedule_type == "cosineannealinglr":
            main_lr_scheduler = ch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - lr_warmup_epochs, eta_min=lr_min
            )

        elif lr_schedule_type == "exponentiallr":
            main_lr_scheduler = ch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=lr_gamma
            )

        if lr_warmup_epochs > 0:
            if lr_warmup_method == "linear":
                warmup_lr_scheduler = ch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=lr_warmup_decay,
                    total_iters=lr_warmup_epochs,
                )

            lr_scheduler = ch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                milestones=[lr_warmup_epochs],
            )

        else:
            lr_scheduler = main_lr_scheduler

        self.scheduler = lr_scheduler

    @param("data.train_dataset")
    @param("data.num_workers")
    @param("training.batch_size")
    @param("training.distributed")
    @param("data.in_memory")
    @param("training.auto_augment")
    @param("training.random_erase_prob")
    def create_train_loader(
        self,
        train_dataset,
        num_workers,
        batch_size,
        distributed,
        in_memory,
        auto_augment,
        random_erase_prob,
    ):
        this_device = f"cuda:{self.gpu}"
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        if auto_augment:
            image_pipeline: List[Operation] = [
                self.decoder,
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(ch.device(this_device), non_blocking=True),
                ToTorchImage(),
                # T.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
                T.RandomErasing(p=random_erase_prob),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            ]
        else:
            image_pipeline: List[Operation] = [
                self.decoder,
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(ch.device(this_device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            ]

        label_pipeline: List[Operation] = [
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

    @param("data.val_dataset")
    @param("data.num_workers")
    @param("validation.batch_size")
    @param("validation.resolution")
    @param("training.distributed")
    def create_val_loader(
        self, val_dataset, num_workers, batch_size, resolution, distributed
    ):
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

    @param("training.epochs")
    @param("logging.log_level")
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {"train_loss": train_loss, "epoch": epoch}

                self.eval_and_log(epoch=epoch, extra_dict=extra_dict)

            self.scheduler.step()

        self.eval_and_log(extra_dict={"epoch": epoch})
        if self.gpu == 0:
            ch.save(self.model.state_dict(), self.log_folder / "final_weights.pt")

    @param("logging.every_n_epochs")
    def eval_and_log(self, epoch: int = 0, every_n_epochs: int = 5, extra_dict={}):
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

    @param("model.arch")
    # @param("model.weights")
    @param("training.distributed")
    @param("training.use_blurpool")
    @param("training.task")
    def create_model_and_scaler(self, arch, distributed, use_blurpool, task):
        scaler = GradScaler()
        model = getattr(models, arch)(pretrained=None)

        def apply_blurpool(mod: ch.nn.Module):
            for name, child in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (
                    np.max(child.stride) > 1 and child.in_channels >= 16
                ):
                    setattr(mod, name, BlurPoolConv2d(child))
                else:
                    apply_blurpool(child)

        if use_blurpool:
            apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    @param("logging.log_level")
    def train_loop(self, epoch, log_level):
        model = self.model
        model.train()
        losses = []

        # lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        # lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            ### Training start
            # for param_group in self.optimizer.param_groups:
            #     param_group["lr"] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = self.model(images)
                loss_train = self.loss(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ["ep", "iter", "shape", "lrs"]
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ["loss"]
                    values += [f"{loss_train.item():.3f}"]

                msg = ", ".join(f"{n}={v}" for n, v in zip(names, values))
                iterator.set_description(msg)
            ### Logging end

            # if ix == 5:
            #     break

        if log_level > 0:
            loss = ch.stack(losses).mean().cpu()
            assert not ch.isnan(loss), "Loss is NaN!"
            return loss.item()

    @param("validation.lr_tta")
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    for k in ["top_1", "top_5"]:
                        self.val_meters[k](output, target)

                    loss_val = self.loss(output, target)
                    self.val_meters["loss"](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param("logging.folder")
    @param("logging.model_ckpt_path")
    def initialize_logger(self, folder, model_ckpt_path):
        self.val_meters = {
            "top_1": torchmetrics.Accuracy(num_classes=1000).to(self.gpu),
            "top_5": torchmetrics.Accuracy(num_classes=1000, top_k=5).to(self.gpu),
            "loss": MeanScalarMetric().to(self.gpu),
        }

        if self.gpu == 0:
            self.model_ckpt_path = create_version_dir(model_ckpt_path, str(self.uid))
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f"=> Logging in {self.log_folder}")
            params = {
                ".".join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

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
    @param("training.distributed")
    @param("dist.world_size")
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param("training.distributed")
    @param("training.eval_only")
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()


# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("sum", default=ch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=ch.tensor(0), dist_reduce_fx="sum")

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count


# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description="Fast imagenet training")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()


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


if __name__ == "__main__":
    # parser = ArgumentParser(description="Fast imagenet training")
    # parser.add_argument("config_file", type=str, help="Path to the config file")
    # args = parser.parse_args()
    make_config(quiet=False)
    ImageNetTrainer.launch_from_args()
