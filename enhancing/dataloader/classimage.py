# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from DALLE-pytorch (https://github.com/lucidrains/DALLE-pytorch)
# Copyright (c) 2020 Phil Wang. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import Optional, Union, Callable, Tuple, Any
from pathlib import Path
from random import randint, choice
from omegaconf import OmegaConf
import PIL

import torch
from torchvision import transforms as T
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
        transform = T.Compose([
            T.RandomResizedCrop(resolution, scale=(resize_ratio, 1.), ratio=(1., 1.)),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        
        super().__init__(root, 'train', transform)


class ClassImageValidation(ClassImageBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
            
        transform = T.Compose([
            T.Resize(resolution),
            T.ToTensor()
        ])
        
        super().__init__(root, 'val', transform)
