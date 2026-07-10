#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import norm  # type: ignore[import-untyped]
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import torchvision  # type: ignore[import-untyped]
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent

np.random.seed(0)
torch.manual_seed(0)



print(f"# 1 : Tokenizer")

class CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([chr(i) for i in ids])


#chars = ["h", "e", "l", "l", "o"]
#print("".join(chars))
#print("-".join(chars))
#
#tokenizer = CharTokenizer()
#text = "Hello 世界 🌏️"
#
#ids = tokenizer.encode(text)
#print(f"ids = {ids}")
#
#decoded = tokenizer.decode(ids)
#print(f"decoded = {decoded}")


#encoded = "A".encode("utf-8")
#print(f"encoded = {encoded!r}")
#print(f"list(encoded) = {list(encoded)}")
#
#ids = [65]
#decoded = bytes(ids).decode("utf-8")
#print(f"decoded = {decoded}")
#
#encoded = "あ".encode("utf-8")
#print(f"encoded = {encoded!r}")
#print(f"list(encoded) = {list(encoded)}")


class ByteTokenizer:
    def encode(self, text: str) -> bytes:
        return text.encode("utf-8")

    def decode(self, ids: bytes) -> str:
        return bytes(ids).decode("utf-8")


#tokenizer = ByteTokenizer()
#text = "Hello 世界 🌏️"
#
#ids = tokenizer.encode(text)
#print(f"ids = {list(ids)}")
#
#decoded = tokenizer.decode(ids)
#print(f"decoded = {decoded}")



def count_pairs(ids: list[int]) -> dict[tuple[int,int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)  # default = 0
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


ids = [1, 2, 3, 1, 2]
counts = count_pairs(ids)
print(f"counts = {counts}")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged_ids: list[int] = []

    i: int = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            # same as pair
            merged_ids.append(new_id)
            i += 2

        else:
            # single char
            merged_ids.append(ids[i])
            i += 1

    return merged_ids


ids = [1, 2, 3, 1, 2]
merged = merge(ids, (1, 2), 4)
print(f"merged = {merged}")
