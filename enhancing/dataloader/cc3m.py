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
from torch.utils.data import Dataset
from torchvision import transforms as T

from ..utils.general import initialize_from_config

class CC3MBase(Dataset):
    def __init__(self, folder: str, split: str,
                 tokenizer: OmegaConf,
                 transform: Callable) -> None:
        super().__init__()

        for line in open(f'{Path(folder)}/{split}_list.txt', 'r').readlines():
            imgpath, text = line.strip().split('\t')
            self.items.append((Path(folder)/imgpath, text))

        self.tokenizer = initialize_from_config(tokenizer)
        self.image_transform = transform

    def __len__(self) -> int:
        return len(self.keys)

    def random_sample(self) -> Tuple[Any, Any]:
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind: int) -> Tuple[Any, Any]:
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind: int) -> Tuple[Any, Any]:
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind: int) -> Tuple[Any, Any]:
        image_file, text_file = self.items[ind]
                
        tokenized_text = self.tokenizer.tokenize(description).squeeze(0)
        image_tensor = Image.open(imgpath).convert('RGB')
        if self.transform:
            image_tensor = self.transform(img)

        # Success
        return {"caption": tokenized_text, "image": image_tensor}


class CC3MTrain(TextImageBase):
    def __init__(self, folder: str,
                 tokenizer: OmegaConf,
                 resolution: Union[Tuple[int, int], int] = 256,
                 resize_ratio: float = 0.75) -> None:
        transform = T.Compose([
            T.RandomResizedCrop(resolution, scale=(resize_ratio, 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])
        
        super().__init__(folder, 'train', tokenizer, transform)


class CC3MValidation(TextImageBase):
    def __init__(self, folder: str,
                 tokenizer: OmegaConf,
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
            
        transform = T.Compose([
            T.Resize(resolution),
            T.ToTensor()
        ])
        
        super().__init__(folder, 'val', tokenizer, transform)
