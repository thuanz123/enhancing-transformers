# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import PIL
from typing import Any, Tuple, Union

import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


class ImageNetBase(ImageFolder):
    def __init__(self, root: str, split: str,
                 transform: Optional[Callable] = None) -> None:
        root = Path(root)/'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC'/split
        super().__init__(root=root, transform=transform)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)

        return {'image': image, 'class': torch.tensor([target])}


class ImageNetTrain(ImageNetBase):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        
        super().__init__(root=root, split='train',  transform=transform)


class ImageNetValidation(ImageNetBase):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor()
        ])
        
        super().__init__(root=root, split='val',  transform=transform)
