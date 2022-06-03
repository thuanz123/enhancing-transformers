
import PIL
from typing import Any, Tuple, Union
from pathlib import Path
from typing import Optional, Union, Callable, Tuple, Any

import torch
import os 
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets import INaturalist


class NaturalTrain(INaturalist):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        version  = '2021_train_mini'
        if not os.path.isdir('data/'+version): 
            super().__init__(root=root, version=version, transform=transform,download=True)
        else:
            super().__init__(root=root, version=version, transform=transform,download=False)

        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)
        return {'image': image, 'class': torch.tensor([target])}
       

class NaturalValidation(INaturalist):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor()
        ])
        
        version  = '2021_valid'
        if not os.path.isdir('data/'+version): 
            super().__init__(root=root, version=version, transform=transform,download=True)
        else:
            super().__init__(root=root, version=version, transform=transform,download=False)

                
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)
        return {'image': image, 'class': torch.tensor([target])}
