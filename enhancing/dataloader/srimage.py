# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from DALLE-pytorch (https://github.com/lucidrains/DALLE-pytorch)
# Copyright (c) 2020 Phil Wang. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import Optional, Tuple, Callable, Union
from pathlib import Path
from random import randint, choice
from omegaconf import OmegaConf
import PIL

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T


class SRBase(Dataset):
    def __init__(self, folder: str, split: str, transform: Callable) -> None:
        super().__init__()
        path = Path(folder)/split

        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]
        
        image_files = {image_file.stem: image_file for image_file in image_files}
        keys = image_files.keys()

        self.keys = list(keys)
        self.image_files = {k: v for k, v in image_files.items() if k in keys}

        self.hr_transform = transform
        
    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        return self.sequential_sample(ind=ind)

    def pad(self, img: PIL.Image.Image) ->  PIL.Image.Image:
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)

        assert img.size[0] <= self.resolution[1] and img.size[1] <= self.resolution[0]
        left = (self.resolution[1] - img.size[0]) // 2
        top = (self.resolution[0] - img.size[1]) // 2
        right = self.resolution[1] - img.size[0] - left
        bottom = self.resolution[0] - img.size[1] - top
        
        return T.functional.pad(img, (left, top, right, bottom))

    def __getitem__(self, ind):
        key = self.keys[ind]
        image_file = self.image_files[key]

        try:
            hr_img = PIL.Image.open(image_file)
            if hr_img.mode != 'RGB':
                hr_img = hr_img.convert('RGB')
                
            hr_tensor = self.hr_transform(hr_img)

            down_size = (hr_tensor.shape[1]//self.downscale, hr_tensor.shape[2]//self.downscale)
            lr_tensor = T.Resize(down_size, 3)(hr_tensor)
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return {'low resolution': lr_tensor, 'high resolution': hr_tensor}


class SRTrain(SRBase):
    def __init__(self, folder: str,
                 resolution: Union[Tuple[int, int], int] = 2048,
                 crop_resolution: Union[Tuple[int, int], int] = 512,
                 downscale: int = 4) -> None:
        assert resolution % downscale == 0
        self.resolution = resolution
        self.downscale = downscale

        transform = T.Compose([
            T.RandomCrop(crop_resolution),
            T.Lambda(self.pad),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor()
        ])        
        
        super().__init__(folder, 'train', transform)
        

class SRValidation(SRBase):
    def __init__(self, folder: str,
                 resolution: Union[Tuple[int, int], int] = 2048,
                 downscale: int = 4) -> None:
        assert resolution % downscale == 0
        self.resolution = resolution
        self.downscale = downscale
        
        transform = T.Compose([
            T.Lambda(self.pad),
            T.ToTensor()
        ])

        super().__init__(folder, 'val', transform)
        
