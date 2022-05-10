# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import numpy as np
from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F

from ...utils.general import get_obj_from_str


def VQCond(base_class: str, *args, **kwargs) -> object:
    def to_img(x: torch.FloatTensor) -> torch.FloatTensor:
        return x.clamp(0, 1)

    model = get_obj_from_str(base_class)(*args, **kwargs)
    model.to_img = to_img

    return model


def VQSegmentation(base_class: str, n_labels: int, *args, **kwargs) -> object:
    base_model_cls = get_obj_from_str(base_class)
    class Wrapper(base_model_cls):
        def __init__(self) -> None:
            self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))
            super().__init__(*args, **kwargs)

        def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
            x = self.get_input(batch, self.image_key)
            xrec, qloss = self(x)
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("train/total_loss", total_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
            return aeloss

        def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
            x = self.get_input(batch, self.image_key)
            xrec, qloss = self(x)
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            total_loss = log_dict_ae["val/total_loss"]
            self.log("val/total_loss", total_loss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            return aeloss

        @torch.no_grad()
        def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
            log = dict()
            x = self.get_input(batch, self.image_key).to(self.device)
            xrec, _ = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                # convert logits to indices
                xrec = torch.argmax(xrec, dim=1, keepdim=True)
                xrec = F.one_hot(xrec, num_classes=x.shape[1])
                xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
                x = self.to_img(x)
                xrec = self.to_img(xrec)
            log["inputs"] = x
            log["reconstructions"] = xrec

            return log

        def to_img(self, x: torch.FloatTensor) -> torch.FloatTensor:
            x = F.conv2d(x, weight=self.colorize)
            
            return (x-x.min())/(x.max()-x.min())

    return Wrapper()
