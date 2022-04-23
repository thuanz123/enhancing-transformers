# ------------------------------------------------------------------------------------
# Modified from CLIP (https://github.com/openai/CLIP)
# Copyright (c) 2021 OpenAI. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import ftfy
import html
import regex as re
from pathlib import Path
from functools import lru_cache
from typing import Optional, List, Tuple, Dict, Set

import torch

@lru_cache()
def default_bpe():
    return 'assets/vocab/bpe_simple_vocab_16e6.txt'

@lru_cache()
def bytes_to_unicode() -> Dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word: str) -> List[Tuple[str, str]]:
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer:
    def __init__(self, bpe_path: str = default_bpe(), text_length: int = 256, 
                 truncate_captions: bool = True) -> None:
        self.context_length = text_length
        self.truncate_text = truncate_captions
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = Path(bpe_path).read_text(encoding='utf8').split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        self.vocab_size = 49408

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token: int) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens: List[int], remove_start_end: bool = True, pad_tokens: Optional[Set] = set()) -> str:
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()

        if remove_start_end:
            tokens = [token for token in tokens if token not in (49406, 40407, 0)]
        text = ''.join([self.decoder[token] for token in tokens if token not in pad_tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def tokenize(self, texts: str) -> List[int]:
        if isinstance(texts, str):
            texts = [texts]
        #assert type(texts) == list, f"texts is {texts}"
        all_tokens = [self.encode(text) for text in texts]
        result = torch.zeros(len(all_tokens), self.context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                if self.truncate_text:
                    tokens = tokens[:self.context_length]
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {self.context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
