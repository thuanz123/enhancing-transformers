# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from StyleGAN-Pytorch (https://github.com/lucidrains/stylegan2-pytorch)
# Copyright (c) 2020 Phil Wang. All Rights Reserved.
# ------------------------------------------------------------------------------------


from math import log2, sqrt
from functools import partial
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import filter2d


def hinge_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = - logits_fake.mean() * 2 if logits_real is None else F.relu(1. + logits_fake).mean() 
    loss_real = 0 if logits_real is None else F.relu(1. - logits_real).mean()
    
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = - logits_fake.sigmoid().log().mean() * 2 if logits_real is None else - (1 - logits_fake.sigmoid()).log().mean() 
    loss_real = 0 if logits_real is None else - logits_real.sigmoid().log().mean()
    
    return 0.5 * (loss_real + loss_fake)


def least_square_d_loss(logits_fake: torch.FloatTensor, logits_real: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    loss_fake = logits_fake.pow(2).mean() * 2 if logits_real is None else (1 + logits_fake).pow(2).mean()
    loss_real = 0 if logits_real is None else (1 - logits_real).pow(2).mean() 
    
    return 0.5 * (loss_real + loss_fake)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(self, num_features: int,
                 logdet: Optional[bool] = False,
                 affine: Optional[bool] = True,
                 allow_reverse_init: Optional[bool] = False) -> None:
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input: torch.FloatTensor) -> None:
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input: torch.FloatTensor, reverse: Optional[bool] = False) -> Union[torch.FloatTensor, Tuple]:
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output: torch.FloatTensor) -> torch.FloatTensor:
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
            
        return h


class EqualConv2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int,
                 kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = True,
                 activation: bool = False) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

        self.activation = activation

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        if self.activation:
            out = F.leaky_relu(out, negative_slope=0.2) * 2**0.5

        return out


class EqualLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 bias: bool = True, bias_init: float = 0,
                 lr_mul: float = 1, activation: bool = False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        if self.activation:
            out = F.leaky_relu(out, negative_slope=0.2) * sqrt(2)

        return out


class Blur(nn.Module):
    def __init__(self, blur_kernel: List[int]) -> None:
        super().__init__()
        self.register_buffer('kernel', torch.Tensor(blur_kernel))
        
    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        kernel = self.kernel
        kernel = kernel[None, None, :] * kernel[None, :, None]

        return filter2d(input, kernel, normalized=True)


class StyleBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = EqualConv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            EqualConv2d(input_channels, filters, 3, padding=1, activation=True),
            EqualConv2d(filters, filters, 3, padding=1, activation=True),
        )

        self.downsample = nn.Sequential(
            Blur([1,3,3,1]),
            EqualConv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        x = (x + res) * (1 / sqrt(2))

        return x


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, use_actnorm: bool = False) -> None:
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(weights_init)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        """Standard forward."""
        return self.main(input)


class StyleDiscriminator(nn.Module):
    def __init__(self, image_size:int = 256, network_capacity: int = 16, fmap_max: int = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        filters = [3] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind != (len(chan_in_out) - 1)

            block = StyleBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

        self.stddev_group = 4
        self.stddev_feat = 1
        
        self.final_conv = EqualConv2d(filters[-1]+1, filters[-1], 3, padding=1, activation=True)
        self.final_linear = nn.Sequential(
            EqualLinear(filters[-1] * 2 * 2, filters[-1], activation=True),
            EqualLinear(filters[-1], 1),
        )

    def forward(self, x):
        out = self.blocks(x)
        batch, channel, height, width = out.shape

        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        
        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)
             
        return out.squeeze()
