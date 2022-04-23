# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from LSUN (https://github.com/fyu/lsun)
# Copyright (c) 2015 Fisher Yu. All Rights Reserved.
# ------------------------------------------------------------------------------------


import PIL
from typing import Any, Tuple, Union, List, Optional, Callable
import subprocess
from os.path import join, dirname, abspath, isfile, isdir

import torch
from torchvision import transforms as T
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
                 resolution: Union[Tuple[int, int], int] = 256,
                 resize_ratio: float = 0.75) -> None:

        transform = T.Compose([
            T.RandomResizedCrop(resolution, scale=(resize_ratio, 1.), ratio=(1., 1.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
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
                 resolution: Union[Tuple[int, int], int] = 256,) -> None:

        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        
        transform = T.Compose([
            T.Resize(resolution),
            T.ToTensor(),
        ])

        if classes not in ['train', 'val']:
            if not isinstance(classes, list):
                 classes = [classes]

            classes = [class_+"_val" for class_ in classes]
        else:
            assert classes == 'val'

        super().__init__(root, classes, transform)
