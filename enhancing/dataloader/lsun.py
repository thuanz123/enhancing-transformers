# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
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
        image, target = super().__getitem__(index)

        return {'image': image, 'class': torch.tensor([target])}


class LSUNTrain(LSUNBase):
    def __init__(self, root: str, classes: Union[Tuple[str, str]],
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
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
        transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor()
        ])

        if classes not in ['train', 'val']:
            if not isinstance(classes, list):
                 classes = [classes]

            classes = [class_+"_val" for class_ in classes]
        else:
            assert classes == 'val'

        super().__init__(root, classes, transform)
