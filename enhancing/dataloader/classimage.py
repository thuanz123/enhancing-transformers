# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import Optional, Union, Callable, Tuple, Any
from pathlib import Path
from random import randint, choice
from omegaconf import OmegaConf
import PIL
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torchvision.datasets import ImageFolder

from ..utils.general import initialize_from_config

class ClassImageBase(ImageFolder):
    def __init__(self, root: str, split: str,
                 transform: Callable) -> None:
        root = Path(root)/split
        super().__init__(root, transform)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample, target = super().__getitem__(index)

        return {'image': sample, 'class': torch.tensor([target])}


class ClassImageTrain(ClassImageBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256,
                 resize_ratio: float = 0.75) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = A.Compose([
            A.SmallestMaxSize(max_size=min(resolution)),
            A.RandomCrop(height=resolution[0], width=resolution[1]),
            ToTensorV2()
        ])
        
        super().__init__(root, 'train', transform)


class ClassImageValidation(ClassImageBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = A.Compose([
            A.SmallestMaxSize(max_size=min(resolution)),
            A.CenterCrop(height=resolution[0], width=resolution[1]),
            ToTensorV2()
        ])
        
        super().__init__(root, 'val', transform)
