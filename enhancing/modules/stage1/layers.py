# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from ViT-Pytorch (https://github.com/lucidrains/vit-pytorch)
# Copyright (c) 2020 Phil Wang. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
from typing import Union, Tuple, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Downsample(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, downscale: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=downscale, stride=downscale)

    def forward(self, x) -> torch.FloatTensor:
        x = self.conv(rearrange(x, 'b h w c -> b c h w'))
        x = rearrange(x, 'b c h w -> b h w c')

        return x


class Upsample(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, upscale: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upscale)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x) -> torch.FloatTensor:
        x = self.upsample(rearrange(x, 'b h w c -> b c h w'))
        x = rearrange(self.conv(x), 'b c h w -> b h w c')

        return x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class BiLSTM2D(nn.Module):
    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.lstm_v = nn.LSTM(in_dim, dim, batch_first=True, bidirectional=True)
        self.lstm_h = nn.LSTM(in_dim, dim, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(4 * dim, in_dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        v, _ = self.lstm_v(rearrange(x, 'b h w c -> (b w) h c'))
        v = rearrange(v, '(b w) h c -> b h w c', w=x.shape[2])

        h, _ = self.lstm_h(rearrange(x, 'b h w c -> (b h) w c'))
        h = rearrange(h, '(b h) w c -> b h w c', h=x.shape[1])

        x = torch.cat([v, h], dim=-1)
        x = self.fc(x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            
        return x


class Sequencer(nn.Module):
    def __init__(self, dim: int = 192, hidden_dim: int = 48, mlp_ratio: int = 3, depth: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, BiLSTM2D(dim, hidden_dim)),
                                   PreNorm(dim, FeedForward(dim, dim*mlp_ratio))])
            self.layers.append(layer)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for lstm, ff in self.layers:
            x = lstm(x) + x
            x = ff(x) + x

        return x


class ViTEncoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 3, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.en_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch_embedding(img)
        x += self.en_pos_embedding

        return self.transformer(x)


class ViTDecoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 3, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))        
        self.to_pixel = nn.Sequential(OrderedDict([
            ('reshape', Rearrange('b (h w) c -> b c h w', h=image_height // patch_height)),
            ('upsample', nn.Upsample(scale_factor=(patch_height, patch_width))),
            ('conv_out', nn.Conv2d(dim, channels, kernel_size=3, padding=1))
        ]))
                                      
    def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
        x = self.transformer(token)
        x += self.de_pos_embedding

        return self.to_pixel(x)

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel.conv_out.weight


class SequencerEncoder(nn.Module):
    def __init__(self, mlp_ratio: int = 3, dims: List[int] = [3, 192, 384, 384, 384], hidden_dims: List[int] = [48, 96, 96, 96],
                 stage_depths: List[int] = [4, 3, 8, 3], scales: List[Union[int, Tuple[int, int]]] = [7, 2, 1, 1]) -> None:
        super().__init__()
        assert (len(dims) - 1) == len(hidden_dims) == len(stage_depths) == len(scales)
        
        self.blocks = nn.Sequential()
        for idx in range(len(stage_depths)):
            depth, downscale = stage_depths[idx], scales[idx]
            
            self.blocks.add_module(f'downsample{idx}', Downsample(dims[idx], dims[idx+1], downscale))
            self.blocks.add_module(f'sequencer{idx}', Sequencer(dims[idx+1], hidden_dims[idx], mlp_ratio, depth))

        self.norm = nn.LayerNorm(dims[-1])
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(self.blocks(x))
        
        return x


class SequencerDecoder(nn.Module):
    def __init__(self, mlp_ratio: int = 3, dims: List[int] = [3, 192, 384, 384, 384], hidden_dims: List[int] = [48, 96, 96, 96],
                 stage_depths: List[int] = [4, 3, 8, 3], scales: List[Union[int, Tuple[int, int]]] = [7, 2, 1, 1]) -> None:
        super().__init__()
        assert (len(dims) - 1) == len(hidden_dims) == len(stage_depths) == len(scales)

        self.norm = nn.LayerNorm(dims[-1])
        
        self.blocks = nn.Sequential()
        for idx in reversed(range(len(stage_depths))):
            depth, upscale = stage_depths[idx], scales[idx]
            
            self.blocks.add_module(f'sequencer{idx}', Sequencer(dims[idx+1], hidden_dims[idx], mlp_ratio, depth))
            self.blocks.add_module(f'upsample{idx}', Upsample(dims[idx+1], dims[idx], upscale))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.blocks(self.norm(x))
        x = rearrange(x, 'b h w c -> b c h w')
        
        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.blocks.upsample0.conv.weight
