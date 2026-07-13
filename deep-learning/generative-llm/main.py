#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import norm  # type: ignore[import-untyped]
from typing import Any, Self
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
import re
import regex  # type: ignore[import-untyped]
from collections.abc import Iterable
import pickle

SCRIPT_DIR = Path(__file__).resolve().parent
Path(f"{SCRIPT_DIR}/.tmp").mkdir(parents = True, exist_ok = True)

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



def count_pairs(
        ids: list[int],
        counts: dict[tuple[int, int], int]|None = None,
) -> dict[tuple[int,int], int]:
    if counts is None:
        counts = defaultdict(int)  # default = 0

    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


#ids = [1, 2, 3, 1, 2]
#counts = count_pairs(ids)
#print(f"counts = {counts}")


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


#ids = [1, 2, 3, 1, 2]
#merged = merge(ids, (1, 2), 4)
#print(f"merged = {merged}")



def train_bpe(
        input_text: str,
        target_vocab_size: int,
        end_token = "<|endoftext|>",
) -> dict[tuple[int, int], int]:
    texts: list[str] = input_text.split(end_token)

    ids_list: list[list[int]] = []
    for text in texts:
        for pretoken in pretokenize(text):
            ids_list.append(list(pretoken.encode("utf-8")))

    # 256 : default vocal size (1 byte)
    # 1 : end token
    num_merges: int = target_vocab_size - 256 - 1
    merge_rules: dict[tuple[int, int], int] = {}

    for step in tqdm(range(num_merges), desc = "Training BPE"):
        pair_vs_count: dict[tuple[int,int], int] = defaultdict(int)
        for ids in ids_list:
            pair_vs_count = count_pairs(ids, pair_vs_count)

        if not pair_vs_count:
            # NOP, there is no pair.
            break

        most_available_pair: tuple[int, int] = max(
                pair_vs_count,
                #key = lambda pair: pair_vs_count[pair],
                key = lambda pair: (pair_vs_count[pair], pair[0], pair[1]),
        )

        new_id: int = 256 + step
        merge_rules[most_available_pair] = new_id

        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], most_available_pair, new_id)

    return merge_rules


#text = "Hello world! This is BPE training."
#merge_rules = train_bpe(text, vocab_size = 260)
#print(merge_rules)
#
#print(f"ord('i') = {ord('i')}")
#print(f"ord('s') = {ord('s')}")



class BPETokenizer:
    def __init__(
            self,
            merge_rules: dict[tuple[int, int], int],
            end_token: str = "<|endoftext|>",
    ) -> None:
        self.merge_rules: dict[tuple[int, int], int] = merge_rules
        self.end_token: str = end_token
        self.end_token_id = 256 + len(merge_rules)

        # default token.
        self.id_vs_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # register new bytes pair in merge_rules
        for (id1, id2), new_id in merge_rules.items():
            self.id_vs_bytes[new_id] = self.id_vs_bytes[id1] + self.id_vs_bytes[id2]
        self.id_vs_bytes[self.end_token_id] = self.end_token.encode("utf-8")


        self.vocab_size = len(self.id_vs_bytes)

    def _encode_text(self, text: str) -> list[int]:
        ids: list[int] = list(text.encode("utf-8"))

        # keep merge order
        for pair, new_id in self.merge_rules.items():
            ids = merge(ids, pair, new_id)

        return ids

    def encode(
            self,
            input_text: str,
            show_progress = False,
    ) -> list[int]:
        pattern: str = f"({re.escape(self.end_token)})"

        texts: list[str] = re.split(pattern, input_text)

        all_ids: list[int] = []

        iterator: Iterable[str] = tqdm(texts, desc = "Encoding") if show_progress else texts
        for text in iterator:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize(text):
                    ids: list[int] = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def decode(self, ids: list[int]) -> str:
        bytes_list: list[bytes] = [self.id_vs_bytes[i] for i in ids]

        text_bytes: bytes = b"".join(bytes_list)

        text: str = text_bytes.decode("utf-8", errors = "replace")

        return text

    @staticmethod
    def load_from(filepath: str) -> "BPETokenizer":
        with open(filepath, "rb") as f:
            merge_rules: dict[tuple[int, int], int] = pickle.load(f)
        return BPETokenizer(merge_rules)



#merge_rules = {(115, 32): 256, (105, 256): 257, (105, 110): 258, (258, 258): 259}
#
#tokenizer = BPETokenizer(merge_rules)
#
#text = "Hello World!"
#ids = tokenizer.encode(text)
#decoded = tokenizer.decode(ids)
#print(f"ids = {ids}")
#print(f"decoded = {decoded}")


#sample_text = "Hello world!<|endoftext|>This is BPE training."
#merge_rules = train_bpe(sample_text, target_vocab_size = 260)
#print(f"merge_rules = {merge_rules}")


#merge_rules = {(115, 32): 256, (105, 256): 257, (105, 110): 258, (258, 258): 259}
#tokenizer = BPETokenizer(merge_rules)
#text = "Hello world!<|endoftext|>"
#ids = tokenizer.encode(text)
#decoded = tokenizer.decode(ids)
#print(f"ids = {ids}")
#print(f"decoded = {decoded}")



def pretokenize(text: str) -> list[str]:
    # regex in GPT-2
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return regex.findall(pattern, text)


#text = "Hello! I'm fine."
#pretokens = pretokenize(text)
#print(f"pretokens = {pretokens}")


#sample_text = "Say hello! Why hello? Just hello.<|endoftext|>Good morning!"
#merge_rules = train_bpe(sample_text, target_vocab_size = 280)
#tokenizer = BPETokenizer(merge_rules)
#
#text = "Say hello!"
#ids = tokenizer.encode(text)
#decoded = tokenizer.decode(ids)
#print(f"ids = {ids}")
#print(f"decoded = {decoded}")
#print()
#
#for token_id in ids:
#    print(f"{token_id} -> '{tokenizer.decode([token_id])}'")


#sample_text = "Say hello! Why hello? Just hello.<|endoftext|>Good morning!"
#merge_rules = train_bpe(sample_text, target_vocab_size = 300)
#with open(f"{SCRIPT_DIR}/.tmp/merge_rules.pkl", "wb") as f:
#    pickle.dump(merge_rules, f)
#
#tokenizer = BPETokenizer.load_from(f"{SCRIPT_DIR}/.tmp/merge_rules.pkl")
#ids = tokenizer.encode("Say hello!")
#decoded = tokenizer.decode(ids)
#print(f"ids = {ids}")
#print(f"decoded = {decoded}")



tiny_codes_txt = f"{SCRIPT_DIR}/dataset/codebot/tiny_codes.txt"
#tiny_codes_merge_rules_pkl = f"{SCRIPT_DIR}/.tmp/tiny_codes_merge_rules.pkl"
tiny_codes_merge_rules_pkl = f"{SCRIPT_DIR}/dataset/codebot/tiny_codes_merge_rules.pkl"

#vocab_size = 1000
#text = open(tiny_codes_txt).read()
#merge_rules = train_bpe(text, vocab_size)
#with open(tiny_codes_merge_rules_pkl, "wb") as f:
#    pickle.dump(merge_rules, f)


#tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
#print("first 10 words")
#for token_id in range(256,266):
#    byte_seq = tokenizer.id_vs_bytes[token_id]
#    text = byte_seq.decode("utf-8")
#    print(f"    ID = {token_id}, text = '{text}'")
#print("last 10 words")
#for token_id in range(990, 1000):
#    byte_seq = tokenizer.id_vs_bytes[token_id]
#    text = byte_seq.decode("utf-8")
#    print(f"    ID = {token_id}, text = '{text}'")
#


#tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
#
#text = open(tiny_codes_txt).read()
#ids = tokenizer.encode(text, show_progress = True)
#
tiny_codes_bin = f"{SCRIPT_DIR}/dataset/codebot/tiny_codes.bin"
#
#ids_array = np.array(ids, dtype = np.uint16)
#ids_array.tofile(tiny_codes_bin)
#
#print(f"token id count = {len(ids_array)}")
#print(f"first 20 token id : {ids_array[:20]}")

