# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import Optional, Union, Callable, Tuple, Any
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image

from torchvision import transforms as T
from torch.utils.data import Dataset

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
        self.transform = transform

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, ind: int) -> Tuple[Any, Any]:
        image_file, caption = self.items[ind]
                
        caption = self.tokenizer.tokenize(caption).squeeze(0)

        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Success
        return {"caption": caption, "image": image}


class CC3MTrain(TextImageBase):
    def __init__(self, folder: str, tokenizer: OmegaConf,
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.ToTensor(),
        ])
        
        super().__init__(folder, 'train', tokenizer, transform)


class CC3MValidation(TextImageBase):
    def __init__(self, folder: str, tokenizer: OmegaConf,
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
        ])
        
        super().__init__(folder, 'val', tokenizer, transform)
