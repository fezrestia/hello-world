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
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import json

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



print(f"# 2 : Model")

## key: tuple[float, ...]
## value : float
#movie_preferences = {
#    (8.0, 2.0, 3.0): 85.0,
#    (3.0, 9.0, 1.0): 70.0,
#    (1.0, 2.0, 9.0): 60.0,
#    (5.0, 5.0, 5.0): 75.0,
#    (7.0, 6.0, 2.0): 80.0,
#    (2.0, 7.0, 6.0): 65.0,
#    (9.0, 1.0, 1.0): 90.0,
#}
#
#new_movie = (6.0, 4.0, 5.0)
#
#
#def soft_dictionary(
#        query: tuple[float, ...],
#        dictionary: dict[tuple[float, ...], float],
#) -> tuple[float, tuple[float, ...]]:
#    similarity: list[float] = []
#    for key in dictionary:
#        sim: float = np.dot(query, key)
#        similarity.append(sim)
#
#    exp_similarity: np.ndarray = np.exp(np.asarray(similarity))
#    weights: np.ndarray = exp_similarity / np.sum(exp_similarity)
#
#    result: float = 0.0
#    for weight, value in zip(weights, dictionary.values()):
#        result += weight * value
#
#    return (result, tuple(weights))
#
#
#predicted_rating, weights = soft_dictionary(new_movie, movie_preferences)
#print(f"new movie = {new_movie}, predicted rating = {predicted_rating:.2f}")
#print(f"weights :")
#for key, weight in zip(movie_preferences.keys(), weights):
#    print(f"movie = {key} : {weight * 100:.2f} %")


#K: Tensor = torch.tensor([
#        [8.0, 2.0, 3.0],
#        [3.0, 9.0, 1.0],
#        [1.0, 2.0, 9.0],
#        [5.0, 5.0, 5.0],
#        [7.0, 6.0, 2.0],
#        [2.0, 7.0, 6.0],
#        [9.0, 1.0, 1.0],
#], dtype = torch.float32)
#
#V: Tensor = torch.tensor([
#        85.0,
#        70.0,
#        60.0,
#        75.0,
#        80.0,
#        65.0,
#        90.0,
#], dtype = torch.float32)
#
#Q: Tensor = torch.tensor([
#        [6.0, 4.0, 5.0],
#        [2.0, 8.0, 3.0],
#        [4.0, 3.0, 7.0],
#], dtype = torch.float32)


# Attention(Q, K, V) = softmax(QK^T / sqrt(d))V
#
# Q: (m, d)
# K: (n, d)
# V: (n, dv)
# similarity: (m, n)
# weights: (m, n)
# outputs: (m, dv)
def attention(Q: Tensor, K: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
    d: int = Q.shape[1]

    similarity: Tensor = torch.matmul(Q, K.t())
    weights: Tensor = F.softmax(similarity, dim = 1) / np.sqrt(d)
    output: Tensor = torch.matmul(weights, V)

    print(f"Q.shape, K.shape, V.shape = {Q.shape}, {K.shape}, {V.shape}")
    print(f"similarity.shape = {similarity.shape}")
    print(f"weights.shape = {weights.shape}")
    print(f"output.shape = {output.shape}")

    return output, weights


#predicted_ratings, weights = attention(Q, K, V)
#
#for movie, rating in zip(Q, predicted_ratings):
#    print(f"movie = {movie.numpy()}, rating = {rating.item():.2f}")


#d = 10
#num_samples = 10000
#dot_products = []
#scaled_dot_products = []
#for _ in range(num_samples):
#    q = np.random.randn(d)
#    k = np.random.randn(d)
#    dot_product = np.dot(q, k)
#    scaled_dot_product = dot_product / np.sqrt(d)
#    dot_products.append(dot_product)
#    scaled_dot_products.append(scaled_dot_product)
#plt.figure(figsize = (10, 6))
#plt.hist(dot_products, bins = 50, alpha = 0.5, label = "w/o scaling")
#plt.hist(scaled_dot_products, bins = 50, alpha = 0.5, label = "w/ scaling")
#plt.legend()
#plt.show()
#
#print(f"variance dot_product = {np.var(dot_products)}")
#print(f"variance scaled dot product = {np.var(scaled_dot_products)}")



# B : batch size
# C : context size
# E : embed vector dim (word vec dim)
# D : key/query dim
#class Attention(nn.Module):
#    def __init__(self, embed_dim: int, key_dim: int) -> None:
#        super().__init__()
#
#        self.W_q = nn.Linear(embed_dim, key_dim, bias = False)
#        self.W_k = nn.Linear(embed_dim, key_dim, bias = False)
#        self.W_v = nn.Linear(embed_dim, key_dim, bias = False)
#        self.W_o = nn.Linear(key_dim, embed_dim, bias = False)
#
#        self.key_dim: int = key_dim
#
#    # x : (B, C, E)
#    def forward(self, x: Tensor) -> Tensor:
#        Q: Tensor = self.W_q(x)  # Q : (B, C, E) @ (E, D) = (B, C, D)
#        K: Tensor = self.W_k(x)  # K : (B, C, E) @ (E, D) = (B, C, D)
#        V: Tensor = self.W_v(x)  # V : (B, C, E) @ (E, D) = (B, C, D)
#
#        K_t: Tensor = K.transpose(-2, -1)  # K : (B, C, D), swap axis -2, -1 -> K_t : (B, D, C)
#        scores: Tensor = torch.matmul(Q, K_t)  # (B, C, D) @ (B, D, C) = (B, C, C)
#        scores = scores / (self.key_dim ** 0.5)
#
#        B, C, E = x.shape
#        mask: Tensor = torch.tril(torch.ones(C, C, device = scores.device))  # (C, C) triangle low
#        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, C, C)
#
#        weights: Tensor = F.softmax(scores, dim = -1)  # (B, C, C)
#        hidden: Tensor = torch.matmul(weights, V)  # (B, C, C) @ (B, C, D) = (B, C, D)
#
#        output: Tensor = self.W_o(hidden)  # (B, C, D) @ (D, E) = (B, C, E)
#
#        return output
#
#
#attention = Attention(embed_dim = 256, key_dim = 64)
#x = torch.randn(2, 5, 256)
#y = attention(x)
#print(f"x.shape = {x.shape}")
#print(f"y.shape = {y.shape}")




# B : batch size
# C : context size
# E : embed vector dim (word vec dim)
# H : head count
# D : key/query dim for each head
# V : vocab size

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            head_count: int,
            head_dim: int,
            dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.head_count: int = head_count
        self.head_dim: int = head_dim

        E: int = embed_dim
        H: int = head_count
        D: int = head_dim

        self.W_q = nn.Linear(E, H * D, bias = False)
        self.W_k = nn.Linear(E, H * D, bias = False)
        self.W_v = nn.Linear(E, H * D, bias = False)
        self.W_o = nn.Linear(H * D, E, bias = False)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    # x : (B, C, E)
    def forward(self, x: Tensor) -> Tensor:
        B, C, E = x.shape
        H = self.head_count
        D = self.head_dim

        Q: Tensor = self.W_q(x)  # Q : (B, C, E) @ (E, H * D) = (B, C, H * D)
        K: Tensor = self.W_k(x)  # K : (B, C, E) @ (E, H * D) = (B, C, H * D)
        V: Tensor = self.W_v(x)  # V : (B, C, E) @ (E, H * D) = (B, C, H * D)

        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        K_t: Tensor = K.transpose(-2, -1)  # (B, H, C, D) -> (B, H, D, C)
        scores: Tensor = torch.matmul(Q, K_t)  # (B, H, C, D) @ (B, H, D, C) = (B, H, C, C)
        scores = scores / (D ** 0.5)

        mask: Tensor = torch.tril(torch.ones(C, C, device = scores.device))  # (C, C) triangle low
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, H, C, C)

        weights: Tensor = F.softmax(scores, dim = -1)  # (B, H, C, C)
        weights = self.attention_dropout(weights)

        hidden: Tensor = torch.matmul(weights, V)  # (B, H, C, C) @ (B, H, C, D) = (B, H, C, D)
        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H * D)  # (B, C, H * D)

        output: Tensor = self.W_o(hidden)  # (B, C, H * D) @ (H * D, E) = (B, C, E)
        output = self.output_dropout(output)

        return output


#embed_dim = 512
#head_count = 8
#head_dim = 64
#
#mha = MultiHeadAttention(embed_dim, head_count, head_dim)
#
#batch_size = 2
#context_len = 10
#x = torch.randn(batch_size, context_len, embed_dim)
#
#output = mha(x)
#print(f"x.shape = {x.shape}")
#print(f"output.shape = {output.shape}")


class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    # x : (B, C, E)
    # return : (B, C, E)
    def forward(self, x: Tensor) -> Tensor:
        mean: Tensor = x.mean(dim = -1, keepdim = True)  # (B, C, 1)
        var: Tensor = x.var(dim = -1, keepdim = True, unbiased = False)  # (B, C, 1)
        norm_x: Tensor = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, E)
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    # x : (B, C, E)
    # return : (B, C, E)
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (
                1.0 + torch.tanh(
                        torch.sqrt(torch.tensor(2.0 / torch.pi))
                        * (x + 0.044715 * torch.pow(x, 3))
                )
        )


class FFN(nn.Module):
    def __init__(
            self,
            x_dim: int,
            hidden_dim: int|None = None,
            dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int(4 * x_dim)

        self.layers = nn.Sequential(
                nn.Linear(x_dim, hidden_dim),
                GELU(),
                nn.Linear(hidden_dim, x_dim),
                nn.Dropout(dropout_rate),
        )

    # x : (B, C, E)
    # return : (B, C, E)
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Block(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            head_count: int,
            ffn_hidden_dim: int|None = None,
            dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        head_dim: int = embed_dim // head_count

        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, head_count, head_dim, dropout_rate)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_hidden_dim, dropout_rate)

    # x : (B, C, E)
    # return : (B, C, E)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_context_len: int,
            embed_dim: int,
            head_count: int,
            layer_count: int,
            ffn_hidden_dim: int,
            dropout_rate: float,
    ) -> None:
        super().__init__()

        self.vocab_size: int = vocab_size
        self.max_context_len: int = max_context_len
        self.embed_dim: int = embed_dim
        self.head_count: int = head_count
        self.layer_count: int = layer_count
        self.ffn_hidden_dim: int = ffn_hidden_dim
        self.dropout_rate: float = dropout_rate

        # embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)  # V -> E
        self.pos_embed = nn.Embedding(self.max_context_len, self.embed_dim)  # C -> E, learned positional embedding
        self.dropout = nn.Dropout(self.dropout_rate)

        # transformer
        self.blocks = nn.ModuleList([
                Block(
                        self.embed_dim,
                        self.head_count,
                        self.ffn_hidden_dim,
                        self.dropout_rate,
                )
                for _ in range(self.layer_count)
        ])

        # output
        self.norm = nn.LayerNorm(self.embed_dim)
        self.unembed = nn.Linear(self.embed_dim, self.vocab_size)  # (E, V)

        # weight tying
        self.embed.weight = self.unembed.weight

        # weight init for each module in this class.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    # ids : (B, C) input text
    # return : (B, C, V) output vocab probability for each context token
    def forward(self, ids: Tensor) -> Tensor:
        B, C = ids.shape

        # embed
        pos: Tensor = torch.arange(0, C, dtype = torch.long, device = ids.device)  # (C,) [0, 1, 2, ..., C]
        emb: Tensor = self.embed(ids)  # (B, C) -> (B, C, E) via (V, E)
        pos_emb: Tensor = self.pos_embed(pos)  # (C,) -> (C, E)
        x: Tensor = self.dropout(emb + pos_emb)  # (B, C, E)

        # transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # output
        logits: Tensor = self.unembed(x)  # (B, C, E) @ (E, V) = (B, C, V)
        return logits

    def save_to(self, file_path: str) -> None:
        checkpoint = {
                "model_state_dict": self.state_dict(),
                "vocab_size": self.vocab_size,
                "max_context_len": self.max_context_len,
                "embed_dim": self.embed_dim,
                "head_count": self.head_count,
                "layer_count": self.layer_count,
                "ffn_hidden_dim": self.ffn_hidden_dim,
                "dropout_rate": self.dropout_rate,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path: str, device = "cpu") -> Self:
        checkpoint = torch.load(file_path, map_location = device)

        model = cls(
                vocab_size = checkpoint["vocab_size"],
                max_context_len = checkpoint["max_context_len"],
                embed_dim = checkpoint["embed_dim"],
                head_count = checkpoint["head_count"],
                layer_count = checkpoint["layer_count"],
                ffn_hidden_dim = checkpoint["ffn_hidden_dim"],
                dropout_rate = checkpoint["dropout_rate"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model



gpt_model_pretrain_pt = f"{SCRIPT_DIR}/.tmp/gpt_model_pretrain.pt"

#vocab_size = 1000
#max_context_len = 256
#embed_dim = 384
#head_count = 6
#layer_count = 6
#ffn_hidden_dim = 4 * embed_dim
#dropout_rate = 0.1
#
#model = GPT(
#        vocab_size,
#        max_context_len,
#        embed_dim,
#        head_count,
#        layer_count,
#        ffn_hidden_dim,
#        dropout_rate,
#)
#
#dummy_input = torch.randint(0, vocab_size, (1, max_context_len))
#logits = model(dummy_input)
#print(f"output logits.shape = {logits.shape}")
#
#model.save_to(gpt_model_pretrain_pt)



print(f"# 3 : Training")

class TokenDataset(Dataset):
    def __init__(self, token_ids: list[int]|np.ndarray, context_len: int) -> None:
        self.tokens: Tensor = torch.tensor(token_ids, dtype = torch.long)
        self.context_len: int = context_len

    def __len__(self) -> int:
        # return valid input and label count.
        # ids : 0, 1, 2, 3, 4
        # context = 2
        # input : 0-1, 1-2, 2-3, 3-4
        # label : 1-2, 2-3, 3-4
        # len = len(ids) - context (label len)
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x: Tensor = self.tokens[idx:idx + self.context_len]
        y: Tensor = self.tokens[idx + 1:idx + self.context_len + 1]

        return (x, y)


ids = np.fromfile(tiny_codes_bin, dtype = np.uint16)
#print(f"ids len = {len(ids)}")
dataset = TokenDataset(ids, context_len = 256)
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)


#for inputs, labels in dataloader:
#    print(f"input tensor shape = {inputs.shape}")
#    print(f"label tensor shape = {labels.shape}")
#    break


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device: torch.device = get_device()
model_pretrain_pt = f"{SCRIPT_DIR}/.tmp/model_pretrain_pt"



# Pre-Training GPT
#
## hyper params
#context_len = 256
#vocab_size = 1000
#batch_size = 32
#learning_rate = 3e-4
#max_iters = 20000
#embed_dim = 384
#head_count = 6
#layer_count = 6
#ffn_hidden_dim = 4 * embed_dim
#dropout_rate = 0.1
#
#model = GPT(
#        vocab_size,
#        context_len,
#        embed_dim,
#        head_count,
#        layer_count,
#        ffn_hidden_dim,
#        dropout_rate,
#).to(device)
#
#optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
#
#
#losses = []
#data_iter = cycle(dataloader)  # infinite loop
#pbar = tqdm(range(max_iters))
#
#for i in pbar:
#    # (B, C)
#    batch_x, batch_y = next(data_iter)
#    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#
#    # (B, C) -> (B, C, V)
#    logits = model(batch_x)
#
#    # logits : (B, C, V) -> (B * C, V)
#    # batch_y : (B, C) -> (B * C,)
#    # loss is calculated by V num predicted val and 1 label val.
#    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
#
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#
#    losses.append(loss.item())
#
#    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
#
#plt.figure(figsize = (10, 6))
#plt.plot(losses)
#plt.xlabel("iteration")
#plt.ylabel("loss")
#plt.grid(True)
#plt.savefig(f"{SCRIPT_DIR}/.tmp/loss_pretrain.png")
#
#model.save_to(model_pretrain_pt)



@torch.no_grad()
def generate(
        model: GPT,
        tokenizer: BPETokenizer,
        prompt: str,
        max_new_tokens: int = 10000,
        temperature: float = 1.0,
) -> str:
    model.eval()  # evaluation mode

    device = next(model.parameters()).device

    token_ids: list[int] = tokenizer.encode(prompt)
    ids: Tensor = torch.tensor([token_ids], dtype = torch.long, device = device)  # (B, C)  B == 1

    generated_ids: Tensor = ids.clone()  # default

    for _ in range(max_new_tokens):
        # limit context len
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # (B, C, V) -> (B, 1, V) : last context probability
        logits: Tensor = model(ids)[:, -1, :]
        next_id: Tensor
        if temperature == 0.0:
            next_id = logits.argmax(dim = -1, keepdim = True)  # max probability on V
        else:
            probs: Tensor = F.softmax(logits / temperature, dim = -1)  # softmax on V
            next_id = torch.multinomial(probs, num_samples = 1)  # sampling

        if next_id.item() == tokenizer.end_token_id:
            break

        ids = torch.cat((ids, next_id), dim = 1)  # (B, C)
        generated_ids = torch.cat((generated_ids, next_id), dim = 1)  # (B, C)

    generated_text: str = tokenizer.decode(generated_ids[0].tolist())
    return generated_text



#prompt = "def"
#max_new_tokens = 200
#temperature = 1.0
#
#tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
#model = GPT.load_from(gpt_model_pretrain_pt)
#
#for i in range(5):
#    print(f"------------ sample {i + 1} ------------")
#    generated_text = generate(model, tokenizer, prompt, max_new_tokens, temperature)
#    print(generated_text)
#    print()



tiny_codes_sft_json = f"{SCRIPT_DIR}/dataset/codebot/tiny_codes_sft.json"

#tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
#
#with open(tiny_codes_sft_json) as f:
#    data = json.load(f)
#
#item = data[0]
#print(f"data[0] = {item}")
#
## alpaca format
#text = f"### Instruction:\n{item["instruction"]}\n\n### Response:\n{item["response"]}<|endoftext|>"
#print(f"alpaca text = {text}")
#
#ids_list = tokenizer.encode(text)
#print(f"ids = {ids_list}")



class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BPETokenizer, context_len: int) -> None:
        self.tokenizer: BPETokenizer = tokenizer
        self.context_len: int = context_len
        self.samples: list[tuple[list[int], list[int]]] = []

        with open(data_path) as f:
            data = json.load(f)
        for item in data:
            ids: list[int]
            labels: list[int]
            ids, labels = self._create_sample(
                    item["instruction"],
                    item["response"],
            )
            self.samples.append((ids, labels))

    def _create_sample(self, instruction: str, response: str) -> tuple[list[int], list[int]]:
        prompt: str = f"### Instructions:\n{instruction}\n\n### Response:\n"
        answer: str = f"{response}<|endoftext|>"

        # torkenize
        prompt_ids: list[int] = self.tokenizer.encode(prompt)
        answer_ids: list[int] = self.tokenizer.encode(answer)

        # input/label
        input_ids: list[int] = prompt_ids + answer_ids
        label_ids: list[int] = [-100] * len(prompt_ids) + answer_ids

        # shift 1
        input_ids = input_ids[:-1]
        label_ids = label_ids[1:]

        padding_len: int = self.context_len - len(input_ids)
        if padding_len > 0:
            # short input/label, add padding
            input_ids = input_ids + [0] * padding_len
            label_ids = label_ids + [-100] * padding_len
        elif padding_len < 0:
            # too long input/label, cut out
            input_ids = input_ids[:self.context_len]
            label_ids = label_ids[:self.context_len]

        return (input_ids, label_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        ids: list[int]
        labels: list[int]
        ids, labels = self.samples[idx]
        ids_tensor: Tensor = torch.tensor(ids, dtype = torch.long)
        labels_tensor: Tensor = torch.tensor(labels, dtype = torch.long)
        return (ids_tensor, labels_tensor)



# Supervised Fine Tuning
#

gpt_model_sft_pt = f"{SCRIPT_DIR}/.tmp/gpt_model_sft.pt"

# hyper param
context_len = 256
batch_size = 32
learning_rate = 3e-4
max_iters = 500

tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
sft_dataset = SFTDataset(tiny_codes_sft_json, tokenizer, context_len)
dataloader = DataLoader(sft_dataset, batch_size = batch_size, shuffle = True)

model = GPT.load_from(gpt_model_pretrain_pt, device = device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# train
losses = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (B, C, V) -> (B * C, V)
            batch_y.view(-1),  # (B, C) -> (B * C,)
            ignore_index = -100,  # ignore for loss calc
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

plt.figure(figsize = (10, 6))
plt.plot(losses)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.grid(True)
plt.savefig(f"{SCRIPT_DIR}/.tmp/loss_sft.png")

model.save_to(gpt_model_sft_pt)

