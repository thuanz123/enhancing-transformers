# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
from omegaconf import OmegaConf
from typing import Tuple, Union, List, Any

import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont

from ...utils.general import initialize_from_config


class DummyCond(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def encode(self, condition: Any) -> Tuple[Any, Any, Any]:
        return condition, None, condition

    def decode(self, condition: Any) -> Any:
        return condition

    def encode_codes(self, condition: Any) -> Any:
        return condition

    def decode_codes(self, condition: Any) -> Any:
        return condition


class TextCond(DummyCond):
    def __init__(self, image_size: Union[Tuple[int, int], int], tokenizer: OmegaConf) -> None:
        super().__init__()
        self.image_size = image_size
        self.tokenizer = initialize_from_config(tokenizer)

    def to_img(self, texts: torch.LongTensor) -> torch.FloatTensor:
        W, H = self.image_size if isinstance(self.image_size, tuple) else (self.image_size, self.image_size)
        font = ImageFont.truetype(os.path.join(os.getcwd(), "assets", "font", "arial.ttf"), 12)
        
        imgs = []
        for text in texts:
            text = self.tokenizer.decode(text)
            words = text.split()
            length = 0
            
            for idx, word in enumerate(words):
                if length > 27:
                    length = 0
                    word[idx-int(idx>0)] += '\n'

                length += len(word)
                
            img = Image.new("RGBA", (W, H), "white")
            draw = ImageDraw.Draw(img)
            
            w, h = draw.textsize(text, font)
            draw.text(((W-w)/2,(H-h)/2), text, font=font, fill="black", align="center")

            img = img.convert('RGB')
            img = T.ToTensor()(img)
            imgs.append(img)
 
        return torch.stack(imgs, dim=0)


class ClassCond(DummyCond):
    def __init__(self, image_size: Union[Tuple[int, int], int], class_name: Union[str, List[str]]) -> None:
        super().__init__()
        self.img_size = image_size
        if isinstance(class_name, str):
            if class_name.endswith("txt") and os.path.isfile(class_name):
                self.cls_name = open(class_name, "r").read().split("\n")
            elif "." not in class_name and not os.path.isfile(class_name):
                self.cls_name = class_name
        elif isinstance(class_name, list) and isinstance(class_name[0], str):
            self.cls_name = class_name
        else:
            raise Exception("Class file format not supported")
            
    def to_img(self, clss: torch.LongTensor) -> torch.FloatTensor:
        W, H = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
        font = ImageFont.truetype(os.path.join(os.getcwd(), "assets", "font", "arial.ttf"), 12)
        
        imgs = []
        for cls in clss:
            cls_name = self.cls_name[int(cls)] 
            length = 0
                
            img = Image.new("RGBA", (W, H), "white")
            draw = ImageDraw.Draw(img)
            
            w, h = draw.textsize(cls_name, font)
            draw.text(((W-w)/2,(H-h)/2), cls_name, font=font, fill="black", align="center")

            img = img.convert('RGB')
            img = T.ToTensor()(img)
            imgs.append(img)
 
        return torch.stack(imgs, dim=0)
