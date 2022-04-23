# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from typing import Tuple, Generic, Dict

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import Callback


class SetupCallback(Callback):
    def __init__(self, config: OmegaConf, exp_config: OmegaConf, basedir: Path, logdir: str = "log", ckptdir:str = "ckpt") -> None:
        super().__init__()
        self.logdir = basedir / logdir
        self.ckptdir = basedir / ckptdir
        self.config = config
        self.exp_config = exp_config
        
    def on_pretrain_routine_start(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            
            print("Experiment config")
            print(self.exp_config.pretty())

            print("Model config")
            print(self.config.pretty())
            
            
class ImageLogger(Callback):
    def __init__(self, batch_frequency: int, max_images: int, clamp: bool = True, increase_log_steps: bool =True) -> None:
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir: str, split: str, images: Dict,
                  global_step: int, current_epoch: int, batch_idx: int) -> None:
        root = os.path.join(save_dir, "results", split)
        os.makedirs(root, exist_ok=True)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
            
    def log_img(self, pl_module: pl.LightningModule, batch: Tuple[torch.LongTensor, torch.FloatTensor], batch_idx: int, split: str = "train") -> None:
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N].detach().cpu()
                if self.clamp:
                    images[k] = images[k].clamp(0, 1)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx: int) -> bool:
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                           outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule,
                                outputs: Generic, batch: Tuple[torch.LongTensor, torch.FloatTensor],
                                batch_idx: int, dataloader_idx: int):
        self.log_img(pl_module, batch, batch_idx, split="val")


class DummyScheduler:
    def schedule(self, n):
        return n

    def __call__(self, n):
        return self.schedule(n)


class LambdaWarmUpCosineScheduler:
    def __init__(self, warm_up_steps, min_, max_, start, max_decay_steps):
        assert (max_decay_steps >= warm_up_steps)
        
        self.warm_up_steps = warm_up_steps
        self.start = start
        self.min_ = min_
        self.max_ = max_
        self.max_decay_steps = max_decay_steps
        self.last = 0.
        
    def schedule(self, n):
        if n < self.warm_up_steps:
            res = (self.max_ - self.start) / self.warm_up_steps * n + self.start
            self.last = res
            return res
        else:
            t = (n - self.warm_up_steps) / (self.max_decay_steps - self.warm_up_steps)
            t = min(t, 1.0)
            res = self.min_ + 0.5 * (self.max_ - self.min_) * (1 + np.cos(t * np.pi))
            self.last = res
            return res

    def __call__(self, n):
        return self.schedule(n)
