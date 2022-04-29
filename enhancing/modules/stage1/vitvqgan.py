# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import List, Tuple, Dict, Any, Optional
from omegaconf import OmegaConf

import PIL
import torch
import torch.nn as nn
from torchvision import transforms as T
import pytorch_lightning as pl

from .layers import Encoder, Decoder
from .quantizers import *
from ...utils.general import get_obj_from_str, initialize_from_config


class ViTVQ(pl.LightningModule):
    def __init__(self, image_key: str, hparams: OmegaConf, qparams: OmegaConf,
                 loss: OmegaConf, path: Optional[str] = None, ignore_keys: List[str] = list()) -> None:
        super().__init__()
        self.path = path
        self.ignore_keys = ignore_keys 
        self.image_key = image_key
        self.image_size = hparams.image_size
        self.latent_dim = hparams.dim
        
        self.loss = initialize_from_config(loss)
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
        self.quantizer = VectorQuantizer(**qparams)
        self.pre_quant = nn.Linear(self.latent_dim, qparams.embed_dim)
        self.post_quant = nn.Linear(qparams.embed_dim, self.latent_dim)

        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:    
        quant, diff = self.encode(x)
        dec = self.decode(quant)
        
        return dec, diff

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, _ = self.quantizer(h)
        
        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)
        
        return dec

    def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.pre_quant(h)
        _, _, codes = self.quantizer(h)
        
        return codes

    def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantizer.embedding(code)
        quant = self.quantizer.norm(quant)
        
        if self.quantizer.use_residual:
            quant = quant.sum(-2)  
            
        dec = self.decode(quant)
        
        return dec

    def get_input(self, batch: Tuple[Any, Any], key: str = 'image') -> Any:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()

        return x.contiguous()

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencoder
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.decoder.get_last_layer(), split="train")

            self.log("train/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.decoder.get_last_layer(), split="train")
            
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
            return discloss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.decoder.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.decoder.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters())+
                                   list(self.decoder.parameters())+
                                   list(self.pre_quant.parameters())+
                                   list(self.post_quant.parameters())+
                                   list(self.quantizer.parameters()),
                                   lr=lr, betas=(0.5, 0.9), weight_decay=1e-4)
        
        if hasattr(self.loss, 'discriminator'):
            opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=1e-4)
            
            return [opt_ae, opt_disc], []
        else:
            return opt_ae

    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        quant, _ = self.encode(x)
        
        log["originals"] = x
        log["reconstructions"] = self.decode(quant)
        
        return log


class ViTVQGumbel(ViTVQ):
    def __init__(self, temperature_scheduler: OmegaConf,
                 image_key: str, hparams: OmegaConf, qparams: OmegaConf,
                 loss: OmegaConf, path: Optional[str] = None, ignore_keys: List[str] = list()) -> None:
        super().__init__(image_key, hparams, qparams, loss, None, None)

        self.temperature_scheduler = initialize_from_config(temperature_scheduler)
        self.quantizer = GumbelQuantizer(**qparams)

        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int) -> torch.FloatTensor:
        self.quantizer.temperature = self.temperature_scheduler(self.global_step)
        loss = super().training_step(batch, batch_idx, optimizer_idx)
        
        if optimizer_idx == 0:
            self.log("temperature", self.quantizer.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return loss
