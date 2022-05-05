# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

from omegaconf import OmegaConf
from typing import Tuple, Union, List, Any

import clip
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont

from .dummycond import DummyCond
from ...utils.general import initialize_from_config


class ClipTextCond(DummyCond):
    def __init__(self, image_size: Union[Tuple[int, int], int],
                 clip_model: str, tokenizer: OmegaConf) -> None:
        super().__init__()
        self.image_size = image_size
        self.clip_model, _ = clip.load(clip_model, device=device)
        self.tokenizer = initialize_from_config(tokenizer)

    def encode_codes(self, text: torch.LongTensor) -> torch.FloatTensor:
        with torch.no_grad():
            text_features = model.encode_text(text)

        return text_features

    def to_img(self, texts: torch.LongTensor) -> torch.FloatTensor:
        W, H = self.image_size if isinstance(self.image_size, tuple) else (self.image_size, self.image_size)
        font = ImageFont.truetype("arial.ttf", 12)
        
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


class ClipImageCond(DummyCond):
    def __init__(self, clip_model: str) -> None:
        super().__init__()
        self.clip_model, _ = clip.load(clip_model, device=device)

    def encode_codes(self, image: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            image_features = model.encode_image(image)

        return image_features

    def to_img(self, image: torch.FloatTensor) -> torch.FloatTensor:
        return image.clamp(0, 1)
