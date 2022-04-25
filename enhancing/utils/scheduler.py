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


class DummyScheduler:
    pass


class FixedScheduler:
    def __init__(self, val):
        self.val = val
        
    def schedule(self, n):
        return self.val

    def __call__(self, n):
        return self.schedule(n)


class ExpontentialDecayScheduler:
    def __init__(self, end, decay_every_step, scale_factor):
        self.decay_every_step = decay_every_step
        self.scale_factor = scale_factor

        self.end = end
        self.current = start
        
    def schedule(self, n):
        if not n % decay_ever_step:
            res = np.exp(-scale_factor*n)
            self.current = max(self.end, res)
            
        return self.current

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
