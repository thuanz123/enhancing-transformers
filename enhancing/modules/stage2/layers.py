# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from minDALL-E (https://github.com/kakaobrain/minDALL-E)
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from minGPT (https://github.com/karpathy/minGPT)
# Copyright (c) 2020 Andrej Karpathy. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
from omegaconf import OmegaConf
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 ctx_len: int,
                 cond_len: int,
                 embed_dim: int,
                 n_heads: int,
                 attn_bias: bool,
                 use_mask: bool = True):
        super().__init__()
        assert embed_dim % n_heads == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim, attn_bias)

        self.n_heads = n_heads
        self.ctx_len = ctx_len
        self.use_mask = use_mask
        if self.use_mask:
            self.register_buffer("mask", torch.ones(ctx_len, ctx_len), persistent=False)
            self.mask = torch.tril(self.mask).view(1, ctx_len, ctx_len)
            self.mask[:, :cond_len, :cond_len] = 1

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ww = torch.zeros(1, 1, embed_dim)
            for i in range(embed_dim):
                ww[0, 0, i] = i / (embed_dim - 1)
        self.time_mix = nn.Parameter(ww)

    def forward(self, x, use_cache=False, layer_past=None):
        B, T, C = x.shape

        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)
        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)
        q = self.query(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)
        v = self.value(x).view(T, B*self.n_heads, C//self.n_heads).transpose(0, 1)  # (B*nh, T, hs)

        if use_cache:
            present = torch.stack([k, v])

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        if use_cache and layer_past is not None:
            # Tensor shape below: (B * nh, 1, hs) X (B * nh, hs, K) -> (B * nh, 1, K)
            att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
            att = F.softmax(att, dim=-1)
            y = torch.bmm(att, v)  # (B*nh, 1, K) X (B*nh, K, hs) -> (B*nh, 1, hs)
        else:
            # Tensor shape below: (B * nh, T, hs) X (B * nh, hs, T) -> (B * nh, T, T)
            att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
            if self.use_mask:
                mask = self.mask if T == self.ctx_len else self.mask[:, :T, :T]
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = torch.bmm(att, v)  # (B*nh, T, T) X (B*nh, T, hs) -> (B*nh, T, hs)
        y = y.transpose(0, 1).contiguous().view(T, B, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        
        if use_cache:
            return y.transpose(0, 1).contiguous(), present  # (T, B, C) -> (B, T, C)
        else:
            return y.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)

class FFN(nn.Module):
    def __init__(self, embed_dim, mlp_bias):
        super().__init__()
        self.p0 = nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias)
        self.p1 = nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias)

    def forward(self, x):
        x = self.p0(x)
        # x = F.gelu(x)
        x = torch.square(torch.relu(x))
        x = self.p1(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 ctx_len: int,
                 cond_len: int,
                 embed_dim: int,
                 n_heads: int,
                 mlp_bias: bool,
                 attn_bias: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadSelfAttention(ctx_len=ctx_len,
                                           cond_len=cond_len,
                                           embed_dim=embed_dim,
                                           n_heads=n_heads,
                                           attn_bias=attn_bias,
                                           use_mask=True)
        self.mlp = FFN(embed_dim=embed_dim, mlp_bias=mlp_bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

    def sample(self, x, layer_past=None):
        attn, present = self.attn(self.ln1(x), use_cache=True, layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x, present


class GPT(nn.Module):
    def __init__(self,
                 vocab_cond_size: int,
                 vocab_img_size: int,
                 embed_dim: int,
                 cond_num_tokens: int,
                 img_num_tokens: int,
                 n_heads: int,
                 n_layers: int,
                 mlp_bias: bool = True,
                 attn_bias: bool = True) -> None:
        super().__init__()
        self.img_num_tokens = img_num_tokens 
        self.vocab_cond_size = vocab_cond_size
        
        # condition token and position embedding 
        self.tok_emb_cond = nn.Embedding(vocab_cond_size, embed_dim)
        self.pos_emb_cond = nn.Parameter(torch.zeros(1, cond_num_tokens, embed_dim))
        
        # input token and position embedding
        self.tok_emb_code = nn.Embedding(vocab_img_size, embed_dim)
        self.pos_emb_code = nn.Parameter(torch.zeros(1, img_num_tokens, embed_dim))

        # transformer blocks
        self.blocks = [Block(ctx_len=cond_num_tokens + img_num_tokens,
                             cond_len=cond_num_tokens,
                             embed_dim=embed_dim,
                             n_heads=n_heads,
                             mlp_bias=mlp_bias,
                             attn_bias=attn_bias) for i in range(1, n_layers+1)]
        self.blocks = nn.Sequential(*self.blocks)

        # head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_img_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                codes: torch.LongTensor,
                conds: torch.LongTensor) -> torch.FloatTensor:
        
        codes = codes.view(codes.shape[0], -1)
        codes = self.tok_emb_code(codes)
        conds = self.tok_emb_cond(conds)
        
        codes = codes + self.pos_emb_code
        conds = conds + self.pos_emb_cond

        x = torch.cat([conds, codes], axis=1).contiguous()
        x = self.blocks(x)
        x = self.layer_norm(x)

        x = x[:, conds.shape[1]-1:-1].contiguous()
        logits = self.head(x)
        
        return logits

    def sample(self,
               conds: torch.LongTensor,
               top_k: Optional[float] = None,
               top_p: Optional[float] = None,
               softmax_temperature: float = 1.0,
               use_fp16: bool = True) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        
        past = codes = logits = None
            
        for i in range(self.img_num_tokens):
            if codes is None:
                codes_ = None
                pos_code = None
            else:
                codes_ = codes.clone().detach()
                codes_ = codes_[:, -1:]
                pos_code = self.pos_emb_code[:, i-1:i, :]
                
            logits_, presents = self.sample_step(codes_, conds, pos_code, use_fp16, past)
            
            logits_ = logits_.to(dtype=torch.float32)
            logits_ = logits_ / softmax_temperature

            presents = torch.stack(presents).clone().detach()
            if past is None:
                past = [presents]
            else:
                past.append(presents)

            if top_k is not None:
                v, ix = torch.topk(logits_, top_k)
                logits_[logits_ < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits_, dim=-1)
            
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_idx_remove_cond = cum_probs >= top_p

                sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
                sorted_idx_remove_cond[..., 0] = 0

                indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / torch.sum(probs, dim=-1, keepdim=True)

            idx = torch.multinomial(probs, num_samples=1).clone().detach()
            codes = idx if codes is None else torch.cat([codes, idx], axis=1)
            logits = logits_ if logits is None else torch.cat([logits, logits_], axis=1)

        del past

        return logits, codes

    def sample_step(self,
                    codes: torch.LongTensor,
                    conds: torch.LongTensor,
                    pos_code: torch.LongTensor,
                    use_fp16: bool = True,
                    past: Optional[torch.FloatTensor] = None) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        
        with autocast(enabled=use_fp16):
            presents = []
            
            if codes is None:
                assert past is None
                conds = self.tok_emb_cond(conds)
                x = conds + self.pos_emb_cond
                
                for i, block in enumerate(self.blocks):
                    x, present = block.sample(x, layer_past=None)
                    presents.append(present)
                x = self.layer_norm(x)
                x = x[:, conds.shape[1]-1].contiguous()
            else:
                assert past is not None
                codes = self.tok_emb_code(codes)
                x = codes + pos_code
                
                past = torch.cat(past, dim=-2)
                for i, block in enumerate(self.blocks):
                    x, present = block.sample(x, layer_past=past[i])
                    presents.append(present)

                x = self.layer_norm(x)
                x = x[:, -1].contiguous()

            logits = self.head(x)
            
            return logits, presents


class RQTransformer(nn.Module):
    def __init__(self,
                 vocab_cond_size: int,
                 vocab_img_size: int,
                 embed_dim: int,
                 cond_num_tokens: int,
                 img_num_tokens: int,
                 depth_num_tokens: int,
                 spatial_n_heads: int,
                 depth_n_heads: int,
                 spatial_n_layers: int,
                 depth_n_layers: int,
                 mlp_bias: bool = True,
                 attn_bias: bool = True) -> None:
        super().__init__()
        self.img_num_tokens = img_num_tokens
        self.depth_num_tokens = depth_num_tokens
        self.vocab_img_size = vocab_img_size
        
        # condition token and position embedding 
        self.tok_emb_cond = nn.Embedding(vocab_cond_size, embed_dim)
        self.pos_emb_cond = nn.Parameter(torch.rand(1, cond_num_tokens, embed_dim))
        
        # spatial token and position embedding
        self.tok_emb_code = nn.Embedding(vocab_img_size, embed_dim)
        self.pos_emb_code = nn.Parameter(torch.rand(1, img_num_tokens, embed_dim))

        # depth position embedding
        self.pos_emb_depth = nn.Parameter(torch.rand(1, depth_num_tokens-1, embed_dim))

        # spatial transformer
        self.spatial_transformer = [Block(ctx_len=cond_num_tokens + img_num_tokens,
                                          cond_len=cond_num_tokens,
                                          embed_dim=embed_dim,
                                          n_heads=spatial_n_heads,
                                          mlp_bias=mlp_bias,
                                          attn_bias=attn_bias) for i in range(1, spatial_n_layers+1)]
        self.spatial_transformer = nn.Sequential(*self.spatial_transformer)

        # depth transformer
        self.depth_transformer = [Block(ctx_len=depth_num_tokens,
                                        cond_len=0,
                                        embed_dim=embed_dim,
                                        n_heads=depth_n_heads,
                                        mlp_bias=mlp_bias,
                                        attn_bias=attn_bias) for i in range(1, depth_n_layers+1)]
        self.depth_transformer = nn.Sequential(*self.depth_transformer)

        # head
        self.ln_spatial = nn.LayerNorm(embed_dim)
        self.ln_depth = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_img_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                codes: torch.LongTensor,
                conds: torch.LongTensor) -> torch.FloatTensor:
        
        codes = codes.view(codes.shape[0], -1, codes.shape[-1])
        codes = self.tok_emb_code(codes)
        conds = self.tok_emb_cond(conds)

        codes_cumsum = codes.cumsum(-1)
        codes_sum = codes_cumsum[..., -1, :]
        
        codes = codes_sum + self.pos_emb_code
        conds = conds + self.pos_emb_cond

        h = torch.cat([conds, codes], axis=1).contiguous()
        h = self.ln_spatial(self.spatial_transformer(h))
        h = h[:, conds.shape[1]-1:-1].contiguous()

        v = codes_cumsum[..., :-1, :] + self.pos_emb_depth
        v = torch.cat([h.unsqueeze(2), v], axis=2).contiguous()

        v = v.view(-1, *v.shape[2:])
        v = self.depth_transformer(v)                  
        logits = self.head(self.ln_depth(v))
        
        return logits

    def sample(self,
               conds: torch.LongTensor,
               top_k: Optional[float] = None,
               top_p: Optional[float] = None,
               softmax_temperature: float = 1.0,
               use_fp16: bool = True) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        
        past = codes = logits = None
        B, T, D, S = conds.shape[0], self.img_num_tokens, self.depth_num_tokens, self.vocab_img_size
            
        for i in range(self.img_num_tokens):
            depth_past = None
            
            if codes is None:
                codes_ = None
                pos_code = None
            else:
                codes_ = codes.clone().detach()
                codes_ = codes_[:, -self.depth_num_tokens:]
                pos_code = self.pos_emb_code[:, i-1:i, :]
                
            hidden, presents = self.sample_spatial_step(codes_, conds, pos_code, use_fp16, past)

            presents = torch.stack(presents).clone().detach()
            if past is None:
                past = [presents]
            else:
                past.append(presents)

            last_len = 0 if codes is None else codes.shape[-1]

            for d in range(self.depth_num_tokens):
                if depth_past is None:
                    codes_ = None
                    pos_depth = None
                else:
                    codes_ = codes.clone().detach()
                    codes_ = codes_[:, last_len:]
                    pos_depth = self.pos_emb_depth[:, d-1:d, :]
                
                logits_, depth_presents = self.sample_depth_step(codes_, hidden, pos_depth, use_fp16, depth_past)

                logits_ = logits_.to(dtype=torch.float32)
                logits_ = logits_ / softmax_temperature

                depth_presents = torch.stack(depth_presents).clone().detach()
                if depth_past is None:
                    depth_past = [depth_presents]
                else:
                    depth_past.append(depth_presents)

                if top_k is not None:
                    v, ix = torch.topk(logits_, top_k)
                    logits_[logits_ < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits_, dim=-1)
                
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_idx_remove_cond = cum_probs >= top_p

                    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
                    sorted_idx_remove_cond[..., 0] = 0

                    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    probs = probs / torch.sum(probs, dim=-1, keepdim=True)

                idx = torch.multinomial(probs, num_samples=1).clone().detach()
                codes = idx if codes is None else torch.cat([codes, idx], axis=1)
                logits = logits_ if logits is None else torch.cat([logits, logits_], axis=1)

            del depth_past

        del past

        codes = codes.view(B, T, D)
        logits = logits.view(B * T, D, S)
        
        return logits, codes

    def sample_spatial_step(self,
                            codes: torch.LongTensor,
                            conds: torch.LongTensor,
                            pos_code: torch.LongTensor,
                            use_fp16: bool = True,
                            past: Optional[torch.Tensor] = None) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        
        with autocast(enabled=use_fp16):
            presents = []

            if codes is None:
                assert past is None
                conds = self.tok_emb_cond(conds)
                x = conds + self.pos_emb_cond
                
                for i, block in enumerate(self.spatial_transformer):
                    x, present = block.sample(x, layer_past=None)
                    presents.append(present)
                x = self.ln_spatial(x)
                x = x[:, conds.shape[1]-1:conds.shape[1]].contiguous()
            else:
                assert past is not None
                codes = self.tok_emb_code(codes)
                x = codes.sum(1, keepdim=True) + pos_code
                
                past = torch.cat(past, dim=-2)
                for i, block in enumerate(self.spatial_transformer):
                    x, present = block.sample(x, layer_past=past[i])
                    presents.append(present)

                x = self.ln_spatial(x)
                x = x[:, -1:].contiguous()
                
            return x, presents

    def sample_depth_step(self,
                          codes: torch.LongTensor,
                          hidden: torch.FloatTensor,
                          pos_depth: torch.LongTensor,
                          use_fp16: bool = True,
                          past: Optional[torch.Tensor] = None) -> Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
        
        with autocast(enabled=use_fp16):
            presents = []

            if codes is None:
                assert past is None
                x = hidden
                
                for i, block in enumerate(self.depth_transformer):
                    x, present = block.sample(x, layer_past=None)
                    presents.append(present)
                x = self.ln_depth(x)
            else:
                assert past is not None
                codes = self.tok_emb_code(codes)
                x = codes.sum(1, keepdim=True) + pos_depth
                
                past = torch.cat(past, dim=-2) 
                for i, block in enumerate(self.depth_transformer):
                    x, present = block.sample(x, layer_past=past[i])
                    presents.append(present)

            x = self.ln_depth(x)    
            x = x[:, -1].contiguous()
            
            logits = self.head(x)   

            return logits, presents
