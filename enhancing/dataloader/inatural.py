import os
import PIL
from typing import Any, Tuple, Union
from pathlib import Path
from typing import Optional, Union, Callable, Tuple, Any

import torch
from torchvision import transforms as T
from torchvision.datasets import INaturalist


class INaturalistBase(INaturalist):
    def __init__(self, root: str, version: str, transform: Callable) -> None:
        path_exist = os.path.isdir(os.path.join(root,version)): 
        super().__init__(root=root, version=version, transform=transform, download=path_exist)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)
        return {'image': image, 'class': torch.tensor([target])}


class INaturalistTrain(INaturalistBase):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        super().__init__(root=root, version='2021_train', transform=transform)

class INaturalistValidation(INaturalistBase):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor()
        ])

        super().__init__(root=root, version='2021_valid', transform=transform)
