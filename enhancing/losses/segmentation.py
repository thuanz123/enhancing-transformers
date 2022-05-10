# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def forward(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction,target)

        return loss, {}


class BCELossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1.):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        bce_loss = F.binary_cross_entropy_with_logits(prediction,target)
        loss = bce_loss + self.codebook_weight*qloss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/bce_loss".format(split): bce_loss.detach().mean(),
               "{}/quant_loss".format(split): qloss.detach().mean()
               }

        return loss, log 
