# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
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


class TextImageBase(Dataset):
    def __init__(self, folder: str, split: str,
                 tokenizer: OmegaConf,
                 transform: Callable) -> None:
        super().__init__()
        path = Path(folder)/split

        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
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
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))

        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
                
        tokenized_text = self.tokenizer.tokenize(description).squeeze(0)
        try:
            image = PIL.Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.image_transform(image)
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return {"caption": tokenized_text, "image": image_tensor}


class TextImageTrain(TextImageBase):
    def __init__(self, folder: str,
                 tokenizer: OmegaConf,
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.ToTensor(),
        ])
        
        super().__init__(folder, 'train', tokenizer, transform)


class TextImageValidation(TextImageBase):
    def __init__(self, folder: str,
                 tokenizer: OmegaConf,
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        if isinstance(resolution, int):
            resolution = [resolution, resolution]

        transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
        ])
        
        super().__init__(folder, 'val', tokenizer, transform)
