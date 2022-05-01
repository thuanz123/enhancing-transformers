# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import PIL
from typing import Any, Tuple, Union, List, Optional, Callable
import subprocess
from os.path import join, dirname, abspath, isfile, isdir
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torchvision.datasets import LSUN


class LSUNBase(LSUN):
    def __init__(self, root: str, classes: Union[Tuple[str, str]],
                 transform: Optional[Callable] = None) -> None:
        super().__init__(root, classes, transform)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample, target = super().__getitem__(index)

        return {'image': sample, 'class': torch.tensor([target])}


class LSUNTrain(LSUNBase):
    def __init__(self, root: str, classes: Union[Tuple[str, str]],
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = A.Compose([
            A.SmallestMaxSize(max_size=min(resolution)),
            A.RandomCrop(height=resolution[0], width=resolution[1]),
            A.RandomHorizontalFlip(),
            ToTensorV2()
        ])

        if classes not in ['train', 'val']:
            if not isinstance(classes, list):
                 classes = [classes]

            classes = [class_+"_train" for class_ in classes]
        else:
            assert classes == 'train'
        
        super().__init__(root, classes, transform)
        

class LSUNValidation(LSUNBase):
    def __init__(self, root: str, classes: Union[Tuple[str, str]],
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = A.Compose([
            A.SmallestMaxSize(max_size=min(resolution)),
            A.CenterCrop(height=resolution[0], width=resolution[1]),
            ToTensorV2()
        ])

        if classes not in ['train', 'val']:
            if not isinstance(classes, list):
                 classes = [classes]

            classes = [class_+"_val" for class_ in classes]
        else:
            assert classes == 'val'

        super().__init__(root, classes, transform)
