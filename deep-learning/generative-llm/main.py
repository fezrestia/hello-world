#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import norm  # type: ignore[import-untyped]
from typing import Any, Self, Iterator, cast, override, overload, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import torchvision  # type: ignore[import-untyped]
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from PIL import Image
from tqdm.auto import tqdm  # type: ignore[import-untyped]
from collections import defaultdict
import re
import regex  # type: ignore[import-untyped]
from collections.abc import Iterable
import pickle
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import json
from multiprocessing import Pool
import shutil
import time
from torch.optim.optimizer import Optimizer
from torch.amp import autocast

SCRIPT_DIR = Path(__file__).resolve().parent
Path(f"{SCRIPT_DIR}/.tmp").mkdir(parents = True, exist_ok = True)

np.random.seed(0)
torch.manual_seed(0)

INT_INF = 1 << 60



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
        weight: int = 1,
        counts: dict[tuple[int, int], int]|None = None,
) -> dict[tuple[int,int], int]:
    if counts is None:
        counts = defaultdict(int)  # default = 0

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
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



#def train_bpe(
#        input_text: str,
#        target_vocab_size: int,
#        end_token = "<|endoftext|>",
#) -> dict[tuple[int, int], int]:
#    texts: list[str] = input_text.split(end_token)
#
#    # count pre-token
#    pretoken_vs_count: dict[str, int] = defaultdict(int)
#    for text in tqdm(texts, desc = "Pretokenizing"):
#        for pretoken in pretokenize(text):
#            pretoken_vs_count[pretoken] += 1
#
#    # pretoken -> id
#    ids_vs_count: dict[tuple[int, ...], int] = {
#            tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_vs_count.items()
#    }
#
#    # 256 : default vocal size (1 byte)
#    # 1 : end token
#    num_merges: int = target_vocab_size - 256 - 1
#    merge_rules: dict[tuple[int, int], int] = {}
#
#    pair_vs_count: dict[tuple[int,int], int] = defaultdict(int)
#    pair_vs_ids: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)  # cache
#    for ids, count in ids_vs_count.items():
#        count_pairs(list(ids), count, pair_vs_count)
#        for pair in zip(ids, ids[1:]):  # [0, 1, 2, 3] and [1, 2, 3] -> (0, 1), (1, 2), (2, 3)
#            pair_vs_ids[pair].add(ids)  # register to cache
#
#    for step in tqdm(range(num_merges), desc = "Training BPE"):
#        if not pair_vs_count:
#            # NOP, there is no pair.
#            break
#
#        most_available_pair: tuple[int, int] = max(
#                pair_vs_count,
#                #key = lambda pair: pair_vs_count[pair],
#                key = lambda pair: (pair_vs_count[pair], pair[0], pair[1]),
#        )
#
#        new_id: int = 256 + step
#        merge_rules[most_available_pair] = new_id
#
#        # get cache and delete
#        affected_ids: set[tuple[int, ...]] = pair_vs_ids[most_available_pair]
#        del pair_vs_ids[most_available_pair]
#
#        for ids in affected_ids:
#            ids_count: int = ids_vs_count[ids]
#            new_ids: list[int] = merge(list(ids), most_available_pair, new_id)
#
#            # update related ids
#            del ids_vs_count[ids]
#            ids_vs_count[tuple(new_ids)] = ids_count
#
#            # update old
#            old_pair_vs_count: dict[tuple[int, int], int] = count_pairs(list(ids))
#            for pair, count in old_pair_vs_count.items():
#                # pair count in ids(pretoken) x ids(pretoken) count in text
#                #     = pair count in same pretoken in text. != total pair count in text
#                pair_vs_count[pair] -= count * ids_count
#                if pair_vs_count[pair] <= 0:
#                    del pair_vs_count[pair]
#                pair_vs_ids[pair].discard(ids)  # delete from cache set
#
#            # update new
#            new_pair_vs_count: dict[tuple[int, int], int] = count_pairs(new_ids)
#            for pair, count in new_pair_vs_count.items():
#                # pair count in ids(pretoken) x ids(pretoken) count in text
#                #     = pair count in same pretoken in text. != total pair count in text
#                pair_vs_count[pair] += count * ids_count
#                pair_vs_ids[pair].add(tuple(new_ids))
#
#    return merge_rules



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

        def get_merge_priority(pair: tuple[int, int]) -> int:
            return self.merge_rules.get(pair, INT_INF)  # inf = lowest priority

        while len(ids) > 1:
            # current pair and count
            pair_vs_count: dict[tuple[int, int], int] = count_pairs(ids)

            most_available_pair: tuple[int, int] = min(
                    pair_vs_count,
                    key = get_merge_priority,
            )  # select most high-priority (earliest learned) pair

            if most_available_pair not in self.merge_rules:
                break

            new_id: int = self.merge_rules[most_available_pair]
            ids = merge(ids, most_available_pair, new_id)

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

    # args: file_path, start, end, cache_dir, chunk_idx
    # return: (cache_file_path, len(ids))
    def _encode_chunk(self, args) -> tuple[str, int]:
        file_path: str
        start: int
        end: int
        cache_dir: str
        chunk_idx: int
        file_path, start, end, cache_dir, chunk_idx = args

        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_byte: bytes = f.read(end - start)
            chunk_text: str = chunk_byte.decode("utf-8", errors = "ignore")

            ids: list[int] = self.encode(chunk_text)

        # save to cache
        cache_file: str = f"{cache_dir}/bpe_ids_chunk_{chunk_idx:08d}.npy"
        np.array(ids, dtype = np.uint16).tofile(cache_file)

        return (cache_file, len(ids))

    def encode_file(
            self,
            file_path: str,
            output_file: str,
            num_processes = 8,
            num_chunks = 64,
            cache_dir = ".cache",
    ) -> int:
        os.makedirs(cache_dir, exist_ok = True)

        try:
            chunk_boundaries: list[int] = find_chunk_boundaries(file_path, num_chunks)
            total_chunks: int = len(chunk_boundaries) - 1

            chunk_info_list = []

            for i in range(total_chunks):
                start: int = chunk_boundaries[i]
                end: int = chunk_boundaries[i + 1]
                chunk_info_list.append((
                        file_path,
                        start,
                        end,
                        cache_dir,
                        i,
                ))

            with Pool(processes = num_processes) as pool:
                cache_results: list[tuple[str, int]] = list(tqdm(
                        pool.imap(self._encode_chunk, chunk_info_list),
                        total = len(chunk_info_list),
                        desc = "Encoding chunks",
                ))

            cache_files: list[str] = [r[0] for r in cache_results]
            token_counts: list[int] = [r[1] for r in cache_results]
            total_tokens: int = sum(token_counts)

            # memmap for output
            dtype = np.uint16
            out_file: np.memmap = np.memmap(
                    output_file,
                    dtype = dtype,
                    mode = "w+",
                    shape = (total_tokens,),
            )

            # write cache to out file
            idx: int = 0
            for cache_file in cache_files:
                chunk_data: np.ndarray = np.fromfile(cache_file, dtype = dtype)
                out_file[idx : idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)

            out_file.flush()
            del out_file

        finally:
            # delete cache
            shutil.rmtree(cache_dir)

        return total_tokens

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



#def pretokenize(text: str) -> list[str]:
def pretokenize(text: str) -> Iterator[str]:
    # regex in GPT-2
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    #return regex.findall(pattern, text)
    for m in regex.finditer(pattern, text):
        yield m.group(0)



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

#class MultiHeadAttention(nn.Module):
#    def __init__(
#            self,
#            embed_dim: int,
#            head_count: int,
#            head_dim: int,
#            dropout_rate: float = 0.1,
#    ) -> None:
#        super().__init__()
#        self.head_count: int = head_count
#        self.head_dim: int = head_dim
#
#        E: int = embed_dim
#        H: int = head_count
#        D: int = head_dim
#
#        self.W_q = nn.Linear(E, H * D, bias = False)
#        self.W_k = nn.Linear(E, H * D, bias = False)
#        self.W_v = nn.Linear(E, H * D, bias = False)
#        self.W_o = nn.Linear(H * D, E, bias = False)
#
#        self.attention_dropout = nn.Dropout(dropout_rate)
#        self.output_dropout = nn.Dropout(dropout_rate)
#
#    # x : (B, C, E)
#    def forward(self, x: Tensor) -> Tensor:
#        B, C, E = x.shape
#        H = self.head_count
#        D = self.head_dim
#
#        Q: Tensor = self.W_q(x)  # Q : (B, C, E) @ (E, H * D) = (B, C, H * D)
#        K: Tensor = self.W_k(x)  # K : (B, C, E) @ (E, H * D) = (B, C, H * D)
#        V: Tensor = self.W_v(x)  # V : (B, C, E) @ (E, H * D) = (B, C, H * D)
#
#        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
#        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
#        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
#
#        K_t: Tensor = K.transpose(-2, -1)  # (B, H, C, D) -> (B, H, D, C)
#        scores: Tensor = torch.matmul(Q, K_t)  # (B, H, C, D) @ (B, H, D, C) = (B, H, C, C)
#        scores = scores / (D ** 0.5)
#
#        mask: Tensor = torch.tril(torch.ones(C, C, device = scores.device))  # (C, C) triangle low
#        scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, H, C, C)
#
#        weights: Tensor = F.softmax(scores, dim = -1)  # (B, H, C, C)
#        weights = self.attention_dropout(weights)
#
#        hidden: Tensor = torch.matmul(weights, V)  # (B, H, C, C) @ (B, H, C, D) = (B, H, C, D)
#        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
#        hidden = hidden.view(B, C, H * D)  # (B, C, H * D)
#
#        output: Tensor = self.W_o(hidden)  # (B, C, H * D) @ (H * D, E) = (B, C, E)
#        output = self.output_dropout(output)
#
#        return output


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


#class Block(nn.Module):
#    def __init__(
#            self,
#            embed_dim: int,
#            head_count: int,
#            ffn_hidden_dim: int|None = None,
#            dropout_rate: float = 0.1,
#    ) -> None:
#        super().__init__()
#
#        head_dim: int = embed_dim // head_count
#
#        self.norm1 = LayerNorm(embed_dim)
#        self.attn = MultiHeadAttention(embed_dim, head_count, head_dim, dropout_rate)
#        self.norm2 = LayerNorm(embed_dim)
#        self.ffn = FFN(embed_dim, ffn_hidden_dim, dropout_rate)
#
#    # x : (B, C, E)
#    # return : (B, C, E)
#    def forward(self, x: Tensor) -> Tensor:
#        x = x + self.attn(self.norm1(x))
#        x = x + self.ffn(self.norm2(x))
#        return x


#class GPT(nn.Module):
#    def __init__(
#            self,
#            vocab_size: int,
#            max_context_len: int,
#            embed_dim: int,
#            head_count: int,
#            layer_count: int,
#            ffn_hidden_dim: int,
#            dropout_rate: float,
#    ) -> None:
#        super().__init__()
#
#        self.vocab_size: int = vocab_size
#        self.max_context_len: int = max_context_len
#        self.embed_dim: int = embed_dim
#        self.head_count: int = head_count
#        self.layer_count: int = layer_count
#        self.ffn_hidden_dim: int = ffn_hidden_dim
#        self.dropout_rate: float = dropout_rate
#
#        # embedding layer
#        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)  # V -> E
#        self.pos_embed = nn.Embedding(self.max_context_len, self.embed_dim)  # C -> E, learned positional embedding
#        self.dropout = nn.Dropout(self.dropout_rate)
#
#        # transformer
#        self.blocks = nn.ModuleList([
#                Block(
#                        self.embed_dim,
#                        self.head_count,
#                        self.ffn_hidden_dim,
#                        self.dropout_rate,
#                )
#                for _ in range(self.layer_count)
#        ])
#
#        # output
#        self.norm = nn.LayerNorm(self.embed_dim)
#        self.unembed = nn.Linear(self.embed_dim, self.vocab_size)  # (E, V)
#
#        # weight tying
#        self.embed.weight = self.unembed.weight
#
#        # weight init for each module in this class.
#        self.apply(self._init_weights)
#
#    def _init_weights(self, module: nn.Module) -> None:
#        if isinstance(module, nn.Linear):
#            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
#            if module.bias is not None:
#                torch.nn.init.zeros_(module.bias)
#        elif isinstance(module, nn.Embedding):
#            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
#
#    # ids : (B, C) input text
#    # return : (B, C, V) output vocab probability for each context token
#    def forward(self, ids: Tensor) -> Tensor:
#        B, C = ids.shape
#
#        # embed
#        pos: Tensor = torch.arange(0, C, dtype = torch.long, device = ids.device)  # (C,) [0, 1, 2, ..., C]
#        emb: Tensor = self.embed(ids)  # (B, C) -> (B, C, E) via (V, E)
#        pos_emb: Tensor = self.pos_embed(pos)  # (C,) -> (C, E)
#        x: Tensor = self.dropout(emb + pos_emb)  # (B, C, E)
#
#        # transformer
#        for block in self.blocks:
#            x = block(x)
#        x = self.norm(x)
#
#        # output
#        logits: Tensor = self.unembed(x)  # (B, C, E) @ (E, V) = (B, C, V)
#        return logits
#
#    def save_to(self, file_path: str) -> None:
#        checkpoint = {
#                "model_state_dict": self.state_dict(),
#                "vocab_size": self.vocab_size,
#                "max_context_len": self.max_context_len,
#                "embed_dim": self.embed_dim,
#                "head_count": self.head_count,
#                "layer_count": self.layer_count,
#                "ffn_hidden_dim": self.ffn_hidden_dim,
#                "dropout_rate": self.dropout_rate,
#        }
#        torch.save(checkpoint, file_path)
#
#    @classmethod
#    def load_from(cls, file_path: str, device = "cpu") -> Self:
#        checkpoint = torch.load(file_path, map_location = device)
#
#        model = cls(
#                vocab_size = checkpoint["vocab_size"],
#                max_context_len = checkpoint["max_context_len"],
#                embed_dim = checkpoint["embed_dim"],
#                head_count = checkpoint["head_count"],
#                layer_count = checkpoint["layer_count"],
#                ffn_hidden_dim = checkpoint["ffn_hidden_dim"],
#                dropout_rate = checkpoint["dropout_rate"],
#        )
#        model.load_state_dict(checkpoint["model_state_dict"])
#        model.to(device)
#
#        return model



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


#ids = np.fromfile(tiny_codes_bin, dtype = np.uint16)
##print(f"ids len = {len(ids)}")
#dataset = TokenDataset(ids, context_len = 256)
#dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)


#for inputs, labels in dataloader:
#    print(f"input tensor shape = {inputs.shape}")
#    print(f"label tensor shape = {labels.shape}")
#    break


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("## use cuda device")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("## use mps device")
        return torch.device("mps")
    else:
        print("## use cpu device")
        return torch.device("cpu")


device: torch.device = get_device()



# Pre-Training GPT
#
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
#plt.show()
#
#model.save_to(gpt_model_pretrain_pt)



#@torch.no_grad()
#def generate(
#        model: GPT,
#        tokenizer: BPETokenizer,
#        prompt: str,
#        max_new_tokens: int = 10000,
#        temperature: float = 1.0,
#) -> str:
#    model.eval()  # evaluation mode
#
#    device = next(model.parameters()).device
#
#    token_ids: list[int] = tokenizer.encode(prompt)
#    ids: Tensor = torch.tensor([token_ids], dtype = torch.long, device = device)  # (B, C)  B == 1
#
#    generated_ids: Tensor = ids.clone()  # default
#
#    for _ in range(max_new_tokens):
#        # limit context len
#        if ids.size(1) > model.max_context_len:
#            ids = ids[:, -model.max_context_len:]
#
#        # (B, C, V) -> (B, 1, V) : last context probability
#        logits: Tensor = model(ids)[:, -1, :]
#        next_id: Tensor
#        if temperature == 0.0:
#            next_id = logits.argmax(dim = -1, keepdim = True)  # max probability on V
#        else:
#            probs: Tensor = F.softmax(logits / temperature, dim = -1)  # softmax on V
#            next_id = torch.multinomial(probs, num_samples = 1)  # sampling
#
#        if next_id.item() == tokenizer.end_token_id:
#            break
#
#        ids = torch.cat((ids, next_id), dim = 1)  # (B, C)
#        generated_ids = torch.cat((generated_ids, next_id), dim = 1)  # (B, C)
#
#    generated_text: str = tokenizer.decode(generated_ids[0].tolist())
#    return generated_text



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

## hyper param
#context_len = 256
#batch_size = 32
#learning_rate = 3e-4
#max_iters = 500
#
#tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
#sft_dataset = SFTDataset(tiny_codes_sft_json, tokenizer, context_len)
#dataloader = DataLoader(sft_dataset, batch_size = batch_size, shuffle = True)
#
#model = GPT.load_from(gpt_model_pretrain_pt, device = device)
#optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
#
## train
#losses = []
#data_iter = cycle(dataloader)
#pbar = tqdm(range(max_iters))
#
#for i in pbar:
#    batch_x, batch_y = next(data_iter)
#    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#
#    logits = model(batch_x)
#    loss = F.cross_entropy(
#            logits.view(-1, logits.size(-1)),  # (B, C, V) -> (B * C, V)
#            batch_y.view(-1),  # (B, C) -> (B * C,)
#            ignore_index = -100,  # ignore for loss calc
#    )
#
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#
#    losses.append(loss.item())
#    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
#
#plt.figure(figsize = (10, 6))
#plt.plot(losses)
#plt.xlabel("iteration")
#plt.ylabel("loss")
#plt.grid(True)
#plt.savefig(f"{SCRIPT_DIR}/.tmp/loss_sft.png")
#plt.show()
#
#model.save_to(gpt_model_sft_pt)



# GRPO group relative policy optimization

gpt_model_grpo_pt = f"{SCRIPT_DIR}/.tmp/gpt_model_grpo.pt"

## hyper param
#learning_rate = 7e-6
#max_iters = 500
#n_update_per_generation = 2  # update count / generated data
#eval_interval = 10
#epsilon = 0.2
#group_size = 8  # sampling generated data count from model
#batch_size = 32
#
#tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
#model = GPT.load_from(gpt_model_sft_pt, device = device)
#optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
#
#old_model = GPT.load_from(gpt_model_sft_pt, device = device)
#old_model.eval()


class GRPODataset(Dataset):
    def __init__(self, tokenizer: BPETokenizer) -> None:
        self.tokenizer: BPETokenizer = tokenizer
        self.data: list[tuple[str, int]] = []

        for i in range(1, 10):
            for j in range(1, 10):
                prompt: str = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                gt: int = i + j
                self.data.append((prompt, gt))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.data[idx]

    def get_batch(
            self,
            prompts: list[str],
            responses: list[str],
            device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []

        for prompt, response in zip(prompts, responses):
            prompt_ids: list[int] = self.tokenizer.encode(prompt)
            response_ids: list[int] = self.tokenizer.encode(response)

            input_ids: list[int] = prompt_ids + response_ids
            mask: list[int] = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(input_ids)
            all_masks.append(mask)

        # padding
        max_len: int = max(len(ids) for ids in all_ids)
        padded_ids: list[list[int]] = []
        padded_masks: list[list[int]] = []
        for ids, mask in zip(all_ids, all_masks):
            pad_len: int = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        ids_tensor: Tensor = torch.tensor(padded_ids, dtype = torch.long, device = device)
        masks_tensor: Tensor = torch.tensor(padded_masks, dtype = torch.float, device = device)

        return (ids_tensor, masks_tensor)


def calculate_reward(ground_truth: int, response: str) -> float:
    try:
        matches = re.findall(r"(-?\d+)", response)
        if matches:
            predicted: int = int(matches[-1])  # last val
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0


#def generate_group(
#        model: GPT,
#        tokenizer: BPETokenizer,
#        prompts: list[str],
#        gts: list[int],
#        group_size: int,
#) -> tuple[list[str], list[str], Tensor]:
#    all_prompts: list[str] = []
#    all_responses: list[str] = []
#    all_advantages: list[Tensor] = []
#
#    for prompt, gt in zip(prompts, gts):
#        responses: list[str] = []
#        for _ in range(group_size):
#            full_text: str = generate(model, tokenizer, prompt, temperature = 1.0)
#            response: str = full_text[len(prompt):]  # cut off prompt
#            responses.append(response)
#
#        # (group_size,)
#        rewards: Tensor = torch.tensor([calculate_reward(gt, r) for r in responses])
#        advantages: Tensor = rewards - rewards.mean()
#
#        for response, advantage in zip(responses, advantages):
#            all_prompts.append(prompt)
#            all_responses.append(response)
#            all_advantages.append(advantage)
#
#    # (group_size * prompts size,)
#    return (all_prompts, all_responses, torch.stack(all_advantages))


#def compute_probs(model: GPT, input_ids: Tensor) -> Tensor:
#    logits: Tensor = model(input_ids)  # (B, C) -> (B, C, V)
#    probs: Tensor = F.softmax(logits[:, :-1, :], dim = -1)  # (B, C - 1, V)
#    labels: Tensor = input_ids[:, 1:]  # (B, C-1)
#
#    token_probs: Tensor = torch.gather(  # (B, C - 1, 1)
#            probs,
#            dim = -1,
#            index = labels.unsqueeze(-1),
#    ).squeeze(-1)  # (B, C - 1)
#
#    return token_probs


#def grpo_loss(
#        model: GPT,
#        old_model: GPT,
#        ids: Tensor,
#        mask: Tensor,
#        advantages: Tensor,
#        epsilon: float = 0.2,
#) -> Tensor:
#    probs: Tensor = compute_probs(model, ids)  # (B, C) -> (B, C, V)
#    with torch.no_grad():
#        old_probs: Tensor = compute_probs(old_model, ids)
#
#    # probability rate per token
#    ratio: Tensor = probs / (old_probs + 1e-8)  # (B, C, V)
#    advantages = advantages.unsqueeze(-1)  # (B, C) -> (B, C, 1)
#
#    # (B, C, V)
#    unclipped: Tensor = ratio * advantages
#    clipped: Tensor = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
#
#    mask = mask[:, 1:]  # for label mask, probs are 1 shifted from ids
#    token_objective: Tensor = torch.min(unclipped, clipped) * mask
#
#    n_samples: int = ids.size(0)  # batch x group
#    return -token_objective.sum() / n_samples



# learning loop

#accuracies = []
#current_accuracy = 0.0
#
#grpo_dataset = GRPODataset(tokenizer)
#dataloader = DataLoader(grpo_dataset, batch_size = batch_size, shuffle = True)
#data_iter = cycle(dataloader)
#
#pbar = tqdm(range(max_iters))
#for i in pbar:
#    # get batch data
#    prompts, gts = next(data_iter)
#
#    # update old model <- latest model
#    old_model.load_state_dict(model.state_dict())
#
#    # generate samples on old model, and calculate reward and advantage
#    all_prompts, all_responses, all_advantages = generate_group(
#            old_model,
#            tokenizer,
#            prompts,
#            gts,
#            group_size,
#    )
#
#    # create batch data for learning
#    grpo_ids, grpo_mask = grpo_dataset.get_batch(all_prompts, all_responses, device)
#    all_advantages = all_advantages.to(device)
#
#    # learning loop
#    for _ in range(n_update_per_generation):
#        optimizer.zero_grad()
#
#        loss = grpo_loss(model, old_model, grpo_ids, grpo_mask, all_advantages, epsilon)
#
#        loss.backward()
#
#        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)  # grad clipping
#
#        optimizer.step()
#
#    # evaluate
#    if i % eval_interval == 0:
#        model.eval()
#
#        correct = 0
#        total = 0
#
#        with torch.no_grad():
#            for prompt, gt in grpo_dataset.data:
#                response = generate(model, tokenizer, prompt, temperature = 0)
#                reward = calculate_reward(gt, response)
#                correct += reward > 0
#                total += 1
#
#        model.train()
#
#        current_accuracy = correct / total * 100
#        accuracies.append(current_accuracy)
#
#    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_accuracy:.2f}%"})
#
#plt.plot(accuracies)
#plt.xlabel("iteration")
#plt.ylabel("accurate")
#plt.grid(True)
#plt.savefig(f"{SCRIPT_DIR}/.tmp/accurate_grpo.png")
#plt.show()
#
#model.save_to(gpt_model_grpo_pt)



# interactive
#

#max_new_tokens = 200
#temperature = 1.0
#
#def format_prompt(user_message):
#    return f"### Instruction:\n{user_message}\n\n### Response:\n"
#
#tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
##model = GPT.load_from(gpt_model_sft_pt, device = device)
#model = GPT.load_from(gpt_model_grpo_pt, device = device)
#
#while True:
#    user_input = input("\nYou: ").strip()
#
#    if not user_input:
#        continue
#
#    if user_input == "exit":
#        exit()
#
#    prompt = format_prompt(user_input)
#    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)
#
#    if "### Response:" in response:
#        response = response.split("### Response:")[-1].strip()
#
#    if "\n" in response:
#        print(f"Bot:\n{response}")
#    else:
#        print(f"Bot: {response}")



print(f"# 4 : Tokenizer Advanced")


#vocab_size = 1000
#text = open(tiny_codes_txt).read()
#merge_rules = train_bpe(text, vocab_size)
#print(f"merge_rules = {len(merge_rules)}")


tiny_stories_train_txt = f"{SCRIPT_DIR}/.tmp/tiny_stories_train.txt"
tiny_stories_valid_txt = f"{SCRIPT_DIR}/.tmp/tiny_stories_valid.txt"


def find_chunk_boundaries(
        file_path: str,
        num_chunks: int,
        end_token = "<|endoftext|>",
) -> list[int]:
    byte_end_token: bytes = end_token.encode("utf-8")

    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size: int = f.tell()
        f.seek(0)

        chunk_size: int = file_size // num_chunks

        # chunk start pos (default)
        chunk_boundaries: list[int] = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)  # last chunk

        read_size: int = 4096

        # scan end token and update boundary
        for bnd_end_idx in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bnd_end_idx]  # start pos
            f.seek(chunk_position)

            while True:
                buffer: bytes = f.read(read_size)

                # file end
                if buffer == b"":
                    chunk_boundaries[bnd_end_idx] = file_size
                    break

                # search end token
                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    # hit
                    chunk_boundaries[bnd_end_idx] = chunk_position + end_position
                    break

                # for next loop
                chunk_position += read_size

    # remove dup and sort
    return sorted(set(chunk_boundaries))


#boundaries = find_chunk_boundaries(tiny_stories_train_txt, num_chunks = 64)
#print(f"len(boundaries) = {len(boundaries)}")
#print(f"boundaries[:5] = {boundaries[:5]}")


#def train_bpe(
#        file_path: str,
#        target_vocab_size: int,
#        end_token = "<|endoftext|>",
#) -> dict[tuple[int, int], int]:
#    chunk_boundaries: list[int] = find_chunk_boundaries(
#            file_path,
#            num_chunks = 64,
#            end_token = end_token,
#    )
#
#    pretoken_vs_count: dict[str, int] = defaultdict(int)
#    with open(file_path, "rb") as f:
#        total_chunks: int = len(chunk_boundaries) - 1
#
#        # count pre-token
#        for chunk_idx in tqdm(range(total_chunks), desc = "Pretokenizing"):
#            start: int = chunk_boundaries[chunk_idx]
#            end: int = chunk_boundaries[chunk_idx + 1]
#
#            # read to mem
#            f.seek(start)
#            chunk_bytes: bytes = f.read(end - start)
#            chunk_text: str = chunk_bytes.decode("utf-8", errors = "ignore")
#
#            # pre-tokenize
#            texts: list[str] = chunk_text.split(end_token)
#            for text in texts:
#                for pretoken in pretokenize(text):
#                    pretoken_vs_count[pretoken] += 1
#
#
#    # pretoken -> id
#    ids_vs_count: dict[tuple[int, ...], int] = {
#            tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_vs_count.items()
#    }
#
#    # 256 : default vocal size (1 byte)
#    # 1 : end token
#    num_merges: int = target_vocab_size - 256 - 1
#    merge_rules: dict[tuple[int, int], int] = {}
#
#    pair_vs_count: dict[tuple[int,int], int] = defaultdict(int)
#    pair_vs_ids: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)  # cache
#    for ids, count in ids_vs_count.items():
#        count_pairs(list(ids), count, pair_vs_count)
#        for pair in zip(ids, ids[1:]):  # [0, 1, 2, 3] and [1, 2, 3] -> (0, 1), (1, 2), (2, 3)
#            pair_vs_ids[pair].add(ids)  # register to cache
#
#    for step in tqdm(range(num_merges), desc = "Training BPE"):
#        if not pair_vs_count:
#            # NOP, there is no pair.
#            break
#
#        most_available_pair: tuple[int, int] = max(
#                pair_vs_count,
#                #key = lambda pair: pair_vs_count[pair],
#                key = lambda pair: (pair_vs_count[pair], pair[0], pair[1]),
#        )
#
#        new_id: int = 256 + step
#        merge_rules[most_available_pair] = new_id
#
#        # get cache and delete
#        affected_ids: set[tuple[int, ...]] = pair_vs_ids[most_available_pair]
#        del pair_vs_ids[most_available_pair]
#
#        for ids in affected_ids:
#            ids_count: int = ids_vs_count[ids]
#            new_ids: list[int] = merge(list(ids), most_available_pair, new_id)
#
#            # update related ids
#            del ids_vs_count[ids]
#            ids_vs_count[tuple(new_ids)] = ids_count
#
#            # update old
#            old_pair_vs_count: dict[tuple[int, int], int] = count_pairs(list(ids))
#            for pair, count in old_pair_vs_count.items():
#                # pair count in ids(pretoken) x ids(pretoken) count in text
#                #     = pair count in same pretoken in text. != total pair count in text
#                pair_vs_count[pair] -= count * ids_count
#                if pair_vs_count[pair] <= 0:
#                    del pair_vs_count[pair]
#                pair_vs_ids[pair].discard(ids)  # delete from cache set
#
#            # update new
#            new_pair_vs_count: dict[tuple[int, int], int] = count_pairs(new_ids)
#            for pair, count in new_pair_vs_count.items():
#                # pair count in ids(pretoken) x ids(pretoken) count in text
#                #     = pair count in same pretoken in text. != total pair count in text
#                pair_vs_count[pair] += count * ids_count
#                pair_vs_ids[pair].add(tuple(new_ids))
#
#    return merge_rules
#
#
#vocab_size = 10000
#merge_rules = train_bpe(tiny_stories_train_txt, vocab_size)
#print(f"len(merge_rules) = {len(merge_rules)}")



def proc_pretoken_chunk(args: tuple[str, int, int, str]) -> dict[str, int]:
    file_path: str
    start: int
    end: int
    end_token: str
    file_path, start, end, end_token = args

    pretoken_vs_count: dict[str, int] = defaultdict(int)

    with open(file_path, "rb") as f:
        # read to mem
        f.seek(start)
        chunk_bytes: bytes = f.read(end - start)
        chunk_text: str = chunk_bytes.decode("utf-8", errors = "ignore")

        # pre-tokenize
        texts: list[str] = chunk_text.split(end_token)
        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_vs_count[pretoken] += 1

    return pretoken_vs_count

def train_bpe(
        file_path: str,
        target_vocab_size: int,
        end_token = "<|endoftext|>",
        num_processes: int = 8,
        num_chunks: int = 64,
) -> dict[tuple[int, int], int]:
    chunk_boundaries: list[int] = find_chunk_boundaries(
            file_path,
            num_chunks = num_chunks,
            end_token = end_token,
    )
    total_chunks: int = len(chunk_boundaries) - 1

    # paralell task
    chunk_info_list = []
    for chunk_idx in range(total_chunks):
        start: int = chunk_boundaries[chunk_idx]
        end: int = chunk_boundaries[chunk_idx + 1]
        chunk_info_list.append((file_path, start, end, end_token))

    # run parallel
    with Pool(processes = num_processes) as pool:
        all_results: list[dict[str, int]] = list(tqdm(
                pool.imap(proc_pretoken_chunk, chunk_info_list),
                total = len(chunk_info_list),
                desc = "Pretokenizing",
        ))

    # merge result
    pretoken_vs_count: dict[str, int] = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_vs_count[pretoken] += count


    # pretoken -> id
    ids_vs_count: dict[tuple[int, ...], int] = {
            tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_vs_count.items()
    }

    # 256 : default vocal size (1 byte)
    # 1 : end token
    num_merges: int = target_vocab_size - 256 - 1
    merge_rules: dict[tuple[int, int], int] = {}

    pair_vs_count: dict[tuple[int,int], int] = defaultdict(int)
    pair_vs_ids: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)  # cache
    for ids, count in ids_vs_count.items():
        count_pairs(list(ids), count, pair_vs_count)
        for pair in zip(ids, ids[1:]):  # [0, 1, 2, 3] and [1, 2, 3] -> (0, 1), (1, 2), (2, 3)
            pair_vs_ids[pair].add(ids)  # register to cache

    for step in tqdm(range(num_merges), desc = "Training BPE"):
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

        # get cache and delete
        affected_ids: set[tuple[int, ...]] = pair_vs_ids[most_available_pair]
        del pair_vs_ids[most_available_pair]

        for ids in affected_ids:
            ids_count: int = ids_vs_count[ids]
            new_ids: list[int] = merge(list(ids), most_available_pair, new_id)

            # update related ids
            del ids_vs_count[ids]
            ids_vs_count[tuple(new_ids)] = ids_count

            # update old
            old_pair_vs_count: dict[tuple[int, int], int] = count_pairs(list(ids))
            for pair, count in old_pair_vs_count.items():
                # pair count in ids(pretoken) x ids(pretoken) count in text
                #     = pair count in same pretoken in text. != total pair count in text
                pair_vs_count[pair] -= count * ids_count
                if pair_vs_count[pair] <= 0:
                    del pair_vs_count[pair]
                pair_vs_ids[pair].discard(ids)  # delete from cache set

            # update new
            new_pair_vs_count: dict[tuple[int, int], int] = count_pairs(new_ids)
            for pair, count in new_pair_vs_count.items():
                # pair count in ids(pretoken) x ids(pretoken) count in text
                #     = pair count in same pretoken in text. != total pair count in text
                pair_vs_count[pair] += count * ids_count
                pair_vs_ids[pair].add(tuple(new_ids))

    return merge_rules



tiny_stories_merge_rules_pkl = f"{SCRIPT_DIR}/dataset/storybot/tiny_stories_merge_rules.pkl"

#if __name__ == "__main__":
#    vocab_size = 10000
#    merge_rules = train_bpe(tiny_stories_train_txt, vocab_size, num_processes = 2)
#
#    with open(tiny_stories_merge_rules_pkl, "wb") as f:
#        pickle.dump(merge_rules, f)


#tokenizer = BPETokenizer.load_from(tiny_stories_merge_rules_pkl)
#
#print("first 10")
#for token_id in range(256, 266):
#    byte_seq = tokenizer.id_vs_bytes[token_id]
#    text = byte_seq.decode("utf-8")
#    print(f"    token_id = {token_id}, text = {text}")
#print("last 10")
#for token_id in range(9990, 10000):
#    byte_seq = tokenizer.id_vs_bytes[token_id]
#    text = byte_seq.decode("utf-8")
#    print(f"    token_id = {token_id}, text = {text}")
#
#sample_text = open(tiny_stories_train_txt).read()[:10000]
#byte_count = len(sample_text.encode("utf-8"))
#ids = tokenizer.encode(sample_text)
#ids_count = len(ids)
#compression_ratio = byte_count / ids_count
#print(f"byte count : {byte_count}")
#print(f"ids_count : {ids_count}")
#print(f"compression_ratio = {compression_ratio}")


#tokenizer = BPETokenizer.load_from(tiny_codes_merge_rules_pkl)
#text = open(tiny_codes_txt).read()
#ids = tokenizer.encode(text, show_progress = True)
#print(f"len(ids) = {len(ids)}")



tiny_stories_train_bin = f"{SCRIPT_DIR}/.tmp/tiny_stories_train.bin"
tiny_stories_valid_bin = f"{SCRIPT_DIR}/.tmp/tiny_stories_valid.bin"



#if __name__ == "__main__":
#    tokenizer = BPETokenizer.load_from(tiny_stories_merge_rules_pkl)
#
#    tokenizer.encode_file(
#            tiny_stories_train_txt,
#            tiny_stories_train_bin,
#            num_processes = 2,
#            num_chunks = 64,
#            cache_dir = f"{SCRIPT_DIR}/.cache",
#    )
#
#    tokenizer.encode_file(
#            tiny_stories_valid_txt,
#            tiny_stories_valid_bin,
#            num_processes = 2,
#            num_chunks = 64,
#            cache_dir = f"{SCRIPT_DIR}/.cache",
#    )



print(f"# 5 : Model Advanced")

class RoPE(nn.Module):
    def __init__(self, theta: float, key_dim: int, max_context_len: int) -> None:
        super().__init__()

        assert key_dim % 2 == 0

        half: int = key_dim // 2

        half_ids: Tensor = torch.arange(0, half)  # (half,)
        inv_freq: Tensor = 1.0 / (theta ** ((2.0 * half_ids) / key_dim))  # (half,)

        # positions : 1-axis : 0, 1, 2, 3, ... , m - 1
        # inv_freq : 1-axis : theta_0, theta_1, theta_2, ... , theta_(k - 1)
        # angles : 2-axis : m * theta_k
        positions: Tensor = torch.arange(max_context_len)  # (max_context_len,)
        angles: Tensor = positions[:, None] * inv_freq[None, :]  # (max_context_len, half)

        cos: Tensor = torch.cos(angles)  # (max_context_len, half) cos(m * theta_k)
        sin: Tensor = torch.sin(angles)  # (max_context_len, half) sin(m * theta_k)

        # register to member field (self.cos/sin_cache)
        self.cos_cache: Tensor
        self.sin_cache: Tensor
        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        batch_size, head_count, context_len, key_dim = x.shape

        input_dtype: torch.dtype = x.dtype
        x = x.float()

        # offset limit
        max_context_len: int = self.cos_cache.size(0)
        if offset + context_len > max_context_len:
            offset = max_context_len - context_len

        cos: Tensor = self.cos_cache[offset:offset + context_len]
        sin: Tensor = self.sin_cache[offset:offset + context_len]

        # separate to odd/even for pair
        x_even: Tensor = x[..., 0::2]
        x_odd: Tensor = x[..., 1::2]

        # rotate
        x_rot_even: Tensor = x_even * cos - x_odd * sin
        x_rot_odd: Tensor = x_even * sin + x_odd * cos

        # merge odd/even
        out: Tensor = torch.stack([x_rot_even, x_rot_odd], dim = -1)  # [0, 1], [2, 3], [4, 5], ...
        out = out.reshape(batch_size, head_count, context_len, key_dim)
        out = out.to(input_dtype)

        return out


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
            rope: RoPE,
    ) -> None:
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

        self.rope: RoPE = rope

        self.k_cache: Tensor|None = None
        self.v_cache: Tensor|None = None
        self.cache_offset: int = 0

    # x : (B, C, E)
    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        B, C, E = x.shape
        H = self.head_count
        D = self.head_dim

        Q: Tensor = self.W_q(x)  # Q : (B, C, E) @ (E, H * D) = (B, C, H * D)
        K: Tensor = self.W_k(x)  # K : (B, C, E) @ (E, H * D) = (B, C, H * D)
        V: Tensor = self.W_v(x)  # V : (B, C, E) @ (E, H * D) = (B, C, H * D)

        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        if use_cache:
            Q = self.rope(Q, self.cache_offset)
            K = self.rope(K, self.cache_offset)
        else:
            Q = self.rope(Q)
            K = self.rope(K)

        if use_cache:
            # check prefill (1st time) or decode (after 2nd)
            is_1st_call: bool
            if self.k_cache is None or self.v_cache is None:
                # 1st call, use all as cache, x includes all prompt.
                is_1st_call = True
                self.k_cache = K
                self.v_cache = V
            else:
                # after 2nd call, add latest result to cache, x includes latest prompt only.
                is_1st_call = False
                self.k_cache = torch.cat([self.k_cache, K], dim = 2)  # cat on C dim
                self.v_cache = torch.cat([self.v_cache, V], dim = 2)  # cat on C dim

            # move cache offset to next token (C is current token len)
            self.cache_offset += C

            K = self.k_cache
            V = self.v_cache

        K_t: Tensor = K.transpose(-2, -1)  # (B, H, C, D) -> (B, H, D, C)
        scores: Tensor = torch.matmul(Q, K_t)  # (B, H, C, D) @ (B, H, D, C) = (B, H, C, C)
        scores = scores / (D ** 0.5)

        # do not use causal mask with kv-cache.
        if not use_cache or is_1st_call:
            mask: Tensor = torch.tril(torch.ones(C, C, device = scores.device))  # (C, C) triangle low
            scores = scores.masked_fill(mask == 0, float("-inf"))  # (B, H, C, C)

        weights: Tensor = F.softmax(scores, dim = -1)  # (B, H, C, C)

        hidden: Tensor = torch.matmul(weights, V)  # (B, H, C, C) @ (B, H, C, D) = (B, H, C, D)
        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H * D)  # (B, C, H * D)

        output: Tensor = self.W_o(hidden)  # (B, C, H * D) @ (H * D, E) = (B, C, E)

        return output

    def clear_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.cache_offset = 0



#embed_dim = 512
#n_head = 8
#head_dim = 64
#theta = 10000
#max_context_len = 1024
#
#rope = RoPE(theta, head_dim, max_context_len)
#mha = MultiHeadAttention(embed_dim, n_head, head_dim, rope = rope)
#
#batch_size = 2
#context_len = 10
#x = torch.randn(batch_size, context_len, embed_dim)
#
#output = mha(x)
#print(f"output.shape = {output.shape}")



def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int|None = None) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int(x_dim * 8 / 3)  # 4 * (2 / 3)

        self.W = nn.Linear(x_dim, hidden_dim, bias = False)
        self.V = nn.Linear(x_dim, hidden_dim, bias = False)
        self.O = nn.Linear(hidden_dim, x_dim, bias = False)

    def forward(self, x: Tensor) -> Tensor:
        a: Tensor = self.W(x)
        b: Tensor = self.V(x)

        gated: Tensor = silu(a) * b

        out: Tensor = self.O(gated)
        return out


class RMSNorm(nn.Module):  # root mean square
    # x: embedded_dim, E of (B, C, E)
    def __init__(self, x: int|tuple[int, ...]) -> None:
        super().__init__()
        self.gamma: Tensor = nn.Parameter(torch.ones(x))
        self.eps: float = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        x2: Tensor = x ** 2
        mean_square: Tensor = x2.mean(dim = -1, keepdim = True)
        root_mean_square: Tensor = torch.sqrt(mean_square + self.eps)
        return self.gamma * x / root_mean_square


class Block(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            head_count: int,
            ffn_hidden_dim: int,
            rope: RoPE,
    ) -> None:
        super().__init__()

        head_dim: int = embed_dim // head_count

        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, head_count, head_dim, rope)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ffn_hidden_dim)

    # x : (B, C, E)
    # return : (B, C, E)
    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        x = x + self.attn(self.norm1(x), use_cache = use_cache)
        x = x + self.ffn(self.norm2(x))
        return x

    def clear_cache(self) -> None:
        self.attn.clear_cache()


class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_context_len: int,
            embed_dim: int,
            head_count: int,
            layer_count: int,
            ffn_hidden_dim: int,
            theta: int,
    ) -> None:
        super().__init__()

        self.vocab_size: int = vocab_size
        self.max_context_len: int = max_context_len
        self.embed_dim: int = embed_dim
        self.head_count: int = head_count
        self.layer_count: int = layer_count
        self.ffn_hidden_dim: int = ffn_hidden_dim
        self.theta: int = theta

        # embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)  # V -> E
        head_dim: int = embed_dim // head_count
        rope: RoPE = RoPE(theta, head_dim, max_context_len)

        # transformer
        self.blocks = nn.ModuleList([
                Block(
                        self.embed_dim,
                        self.head_count,
                        self.ffn_hidden_dim,
                        rope,
                )
                for _ in range(self.layer_count)
        ])

        # output
        self.norm = RMSNorm(self.embed_dim)
        self.unembed = nn.Linear(self.embed_dim, self.vocab_size, bias = False)  # (E, V)

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
    def forward(self, ids: Tensor, use_cache: bool = False) -> Tensor:
        # embed
        x: Tensor = self.embed(ids)  # (B, C) -> (B, C, E) via (V, E)

        # transformer
        for block in self.blocks:
            x = block(x, use_cache = use_cache)
        x = self.norm(x)

        # output
        logits: Tensor = self.unembed(x)  # (B, C, E) @ (E, V) = (B, C, V)
        return logits

    def clear_cache(self) -> None:
        for block in self.blocks:
            block = cast(Block, block)
            block.clear_cache()

    def save_to(self, file_path: str) -> None:
        checkpoint = {
                "model_state_dict": self.state_dict(),
                "vocab_size": self.vocab_size,
                "max_context_len": self.max_context_len,
                "embed_dim": self.embed_dim,
                "head_count": self.head_count,
                "layer_count": self.layer_count,
                "ffn_hidden_dim": self.ffn_hidden_dim,
                "theta": self.theta,
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
                theta = checkpoint["theta"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model



#vocab_size = 10000
#max_context_len = 256
#embed_dim = 384
#n_head = 6
#n_layer = 6
#ff_dim = int(embed_dim * 8 / 3)
#theta = 10000
#
#model = GPT(vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, theta)
#
#batch_size = 8
#dummy_input = torch.randint(0, vocab_size, (batch_size, max_context_len))
#
#logits = model(dummy_input)
#
#print(f"input.shape = {dummy_input.shape}")
#print(f"logits.shape = {logits.shape}")
#
#start_ids = torch.tensor([[42]])
#max_tokens = 200
#
## no cache
#model.clear_cache()
#model.eval()
#start_time = time.time()
#ids = start_ids
#with torch.no_grad():
#    for _ in range(max_tokens):
#        logits = model(ids, use_cache = False)
#        next_id = torch.argmax(logits[:, -1, :], dim = -1, keepdim = True)
#        ids = torch.cat([ids, next_id], dim = 1)
#elapsed = time.time() - start_time
#print(f"elapsed w/o cache = {elapsed}, ids.shape = {ids.shape}")
#
## with cache
#model.clear_cache()
#model.eval()
#start_time = time.time()
#ids = start_ids
#next_id = start_ids
#with torch.no_grad():
#    for _ in range(max_tokens):
#        logits = model(next_id, use_cache = True)
#        next_id = torch.argmax(logits[:, -1, :], dim = -1, keepdim = True)
#        ids = torch.cat([ids, next_id], dim = 1)
#elapsed = time.time() - start_time
#print(f"elapsed w/ cache  = {elapsed}, ids.shape = {ids.shape}")



print(f"# 6 : Learning Advanced")

class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01) -> None:
        defaults: dict[str, Any] = {
                "lr": lr,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @override
    def step(self, closure: Callable[[], float]|None = None) -> float|None:
        for group in self.param_groups:
            lr: float = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data = p.data - lr * p.grad.data

        return None


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[Tensor],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
    ) -> None:
        defaults: dict[str, Any] = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @override
    def step(self, closure: Callable[[], float]|None = None) -> float|None:
        for group in self.param_groups:
            beta1: float
            beta2: float
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad: Tensor = p.grad.data
                state: dict[str, Any] = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t: int = state["t"]

                m: Tensor = state["m"]
                v: Tensor = state["v"]

                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad ** 2

                state["m"] = m
                state["v"] = v

                m_hat: Tensor = m / (1.0 - beta1 ** t)
                v_hat: Tensor = v / (1.0 - beta2 ** t)

                lr: float = group["lr"]
                eps: float = group["eps"]
                wd: float = group["weight_decay"]

                # update param
                p.data = p.data - lr * m_hat / (v_hat.sqrt() + eps) - lr * wd * p.data

        return None



#model = torch.nn.Linear(2, 1)
#optimizer = AdamW(model.parameters(), lr = 0.1)
#
#x = torch.tensor([[1.0, 2.0]])
#y = torch.tensor([[3.0]])
#
#for step in range(5):
#    output = model(x)
#    loss = (output - y).pow(2).mean()
#
#    loss.backward()
#    optimizer.step()
#    optimizer.zero_grad()
#
#    print(f"step = {step}, loss = {loss:.4f}")



def get_learning_rate(
        cur_iters: int,
        warmup_iters: int,
        max_iters: int,
        max_learning_rate: float,
) -> float:
    # warm up
    if cur_iters < warmup_iters:
        return max_learning_rate * (cur_iters / warmup_iters)

    # annealing
    if cur_iters < max_iters:
        progress: float = (cur_iters - warmup_iters) / (max_iters - warmup_iters)
        return max_learning_rate * (1.0 - progress)

    return 0.0



train_data = np.memmap(tiny_stories_train_bin, dtype = np.uint16, mode = "r")



def get_batch(
        data: np.ndarray,
        context_len: int,
        batch_size: int,
        device: torch.device,
        random: bool = True,
        offset: int = 0,
) -> tuple[Tensor, Tensor]:
    batch_start_idx: Tensor
    if random:
        # random int for batch start idx
        batch_start_idx = torch.randint(len(data) - context_len - 1, (batch_size,))
    else:
        # start, end, step, for batch start idx
        batch_start_idx = torch.arange(offset, offset + batch_size * context_len, context_len)
        # cut out overflow, array[bool_array] = array[true only]
        batch_start_idx = batch_start_idx[batch_start_idx + context_len + 1 < len(data)]
        if len(batch_start_idx) == 0:
            raise Exception("Unexpected, batch_start_idx.len == 0")

    x: Tensor = torch.stack([
            torch.from_numpy(data[i:i + context_len].astype(np.int64))
                    for i in batch_start_idx
    ])

    y: Tensor = torch.stack([
            torch.from_numpy(data[i + 1:i + context_len + 1].astype(np.int64))
                    for i in batch_start_idx
    ])

    return (x.to(device), y.to(device))



def evaluate(
        model: GPT,
        val_data: np.ndarray,
        context_len: int,
        batch_size: int,
        device: torch.device,
) -> float:
    model.eval()

    total_loss: float = 0.0
    total_tokens: int = 0

    max_start: int = len(val_data) - context_len - 1
    num_batches: int = (max_start // context_len) // batch_size + 1

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc = "Validation"):
            offset: int = batch_idx * batch_size * context_len
            (x, y) = get_batch(
                    val_data,
                    context_len,
                    batch_size,
                    device,
                    random = False,
                    offset = offset,
            )

            with autocast(device_type = device.type, dtype = torch.bfloat16):
                logits: Tensor = model(x)
                loss: Tensor = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        reduction = "sum",  # only sum, calc mean following with token size
                )

            total_loss += loss.item()
            total_tokens += y.numel()

    model.train()
    return total_loss / total_tokens



# pre-learning

gpt_model_storybot_pretrain_pt = f"{SCRIPT_DIR}/.tmp/gpt_model_storybot_pretrain.pt"

## hyper param
#context_len = 256
#vocab_size = 10000
#batch_size = 32
#learning_rate = 0.001  # max
#warmup_iters = 200
#max_iters = 40000
#embed_dim = 512
#n_head = 16
#n_layer = 4
#ff_dim = 1344
#theta = 10000
#eval_iters = 500
#grad_clip = 1.0
#save_iters = [500, 5000]
#
## load data
#train_data = np.memmap(tiny_stories_train_bin, dtype = np.uint16, mode = "r")
#val_data = np.memmap(tiny_stories_valid_bin, dtype = np.uint16, mode = "r")
#
## components
#tokenizer = BPETokenizer.load_from(tiny_stories_merge_rules_pkl)
#model = GPT(
#        vocab_size,
#        context_len,
#        embed_dim,
#        n_head,
#        n_layer,
#        ff_dim,
#        theta,
#).to(device)
#optimizer = AdamW(model.parameters(), lr = learning_rate)
#
#pbar = tqdm(range(max_iters))
#
#val_loss = float("inf")
#val_losses = []
#val_iters = []
#
#for i in pbar:
#    # update learning rate
#    lr = get_learning_rate(i, warmup_iters, max_iters, learning_rate)
#    for param_group in optimizer.param_groups:
#        param_group["lr"] = lr
#
#    (batch_x, batch_y) = get_batch(train_data, context_len, batch_size, device)
#
#    optimizer.zero_grad()
#
#    with autocast(device_type = device.type, dtype = torch.bfloat16):
#        logits = model(batch_x)
#        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
#
#    loss.backward()
#    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#    optimizer.step()
#
#    # save to
#    if i in save_iters:
#        save_path = f"{SCRIPT_DIR}/.tmp/storybot_model_iter_{i}.pt"
#        model.save_to(save_path)
#        print(f"Model saved at iteration {i}: {save_path}")
#
#    # evaluate
#    if (i % eval_iters) == 0 or i == max_iters - 1:
#        val_loss = evaluate(model, val_data, context_len, batch_size, device)
#        val_losses.append(val_loss)
#        val_iters.append(i)
#
#    pbar.set_postfix({"loss": f"{loss.item():.4f}", "val_loss": f"{val_loss:.6f}"})
#
#model.save_to(gpt_model_storybot_pretrain_pt)
#
#plt.figure(figsize = (10, 6))
#plt.plot(val_iters, val_losses)
#plt.xlabel("iteration")
#plt.ylabel("validation loss")
#plt.grid(True)
#plt.savefig(f"{SCRIPT_DIR}/.tmp/storybot_pretrain_loss_val.png")
#plt.show()



# generate story

@torch.no_grad()
def generate(
        model: GPT,
        tokenizer: BPETokenizer,
        prompt: str,
        max_new_tokens: int = 10000,
        temperature: float = 1.0,
) -> str:
    model.eval()
    model.clear_cache()

    device = next(model.parameters()).device

    token_ids: list[int] = tokenizer.encode(prompt)
    ids: Tensor = torch.tensor([token_ids], dtype = torch.long, device = device)  # (B, C)  B == 1

    generated_ids: Tensor = ids
    next_id: Tensor = ids

    for _ in range(max_new_tokens):
        # limit context len
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # (B, C, V) -> (B, 1, V) : last context probability
        logits: Tensor = model(next_id, use_cache = True)[:, -1, :]
        if temperature == 0.0:
            next_id = logits.argmax(dim = -1, keepdim = True)  # max probability on V
        else:
            probs: Tensor = F.softmax(logits / temperature, dim = -1)  # softmax on V
            next_id = torch.multinomial(probs, num_samples = 1)  # sampling

        if next_id.item() == tokenizer.end_token_id:
            break

        ids = torch.cat((ids, next_id), dim = 1)  # (B, C)
        generated_ids = torch.cat((generated_ids, next_id), dim = 1)  # (B, C)

    # remove end token, use bool index, (B, C) -> (C,)
    generated_ids = generated_ids[generated_ids != tokenizer.end_token_id]

    generated_text: str = tokenizer.decode(generated_ids.tolist())
    return generated_text



#tokenizer = BPETokenizer.load_from(tiny_stories_merge_rules_pkl)
#model = GPT.load_from(gpt_model_storybot_pretrain_pt)
#
#prompt = "<|endoftext|>"
#max_new_tokens = 300
#temperature = 1.0
#num_samples = 3
#
#for i in range(num_samples):
#    story = generate(model, tokenizer, prompt, max_new_tokens, temperature)
#    print(f"------------ story {i} ------------")
#    print(story)
#    print()



class DPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BPETokenizer, context_len: int) -> None:
        self.tokenizer: BPETokenizer = tokenizer
        self.context_len: int = context_len
        self.samples: list[tuple[list[int], list[int], list[int], list[int]]] = []

        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            sample: tuple[list[int], list[int], list[int], list[int]] = self._create_sample(
                    item["prompt"],
                    item["chosen"],
                    item["rejected"],
            )
            self.samples.append(sample)

    def _pad_and_mask(self, ids: list[int], prompt_len: int) -> tuple[list[int], list[int]]:
        mask: list[int] = [0] * prompt_len + [1] * (len(ids) - prompt_len)

        if len(ids) > self.context_len:
            ids = ids[:self.context_len]
            mask = mask[:self.context_len]
        else:
            pad_len: int = self.context_len - len(ids)
            ids = ids + [0] * pad_len
            mask = mask + [0] * pad_len

        return (ids, mask)

    def _create_sample(
            self,
            prompt: str,
            chosen: str,
            rejected: str,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        prompt_ids: list[int] = self.tokenizer.encode(prompt)
        chosen_ids: list[int] = prompt_ids + self.tokenizer.encode(chosen)
        rejected_ids: list[int] = prompt_ids + self.tokenizer.encode(rejected)

        prompt_len: int = len(prompt_ids)
        chosen_mask: list[int]
        rejected_mask: list[int]
        chosen_ids, chosen_mask = self._pad_and_mask(chosen_ids, prompt_len)
        rejected_ids, rejected_mask = self._pad_and_mask(rejected_ids, prompt_len)

        return (chosen_ids, chosen_mask, rejected_ids, rejected_mask)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        chosen_ids, chosen_mask, rejected_ids, rejected_mask = self.samples[idx]

        return (
                torch.tensor(chosen_ids, dtype = torch.long),
                torch.tensor(chosen_mask, dtype = torch.long),
                torch.tensor(rejected_ids, dtype = torch.long),
                torch.tensor(rejected_mask, dtype = torch.long),
        )


# ids/mask includes prompt + chosen/rejected
def get_sequence_logprobs(model: GPT, ids: Tensor, mask: Tensor) -> Tensor:
    logits: Tensor = model(ids)  # (B, C) -> (B, C, V)
    log_probs: Tensor = F.log_softmax(logits[:, :-1, :], dim = -1)  # (B, C - 1, V)
    labels: Tensor = ids[:, 1:]  # (B, C - 1)

    # get each label probability from V
    per_token_logprobs: Tensor = torch.gather(
            log_probs,  # (B, C - 1, V)
            dim = -1,
            index = labels.unsqueeze(-1),  # (B, C - 1, 1)
    ).squeeze(-1)  # (B, C - 1, 1) -> (B, C - 1)

    masked_logprobs: Tensor = per_token_logprobs * mask[:, 1:]  # (B, C - 1)  based on label index

    return masked_logprobs.sum(dim = -1)  # (B,)


def compute_dpo_loss(
        model: GPT,
        ref_model: GPT,
        chosen_ids: Tensor,
        chosen_mask: Tensor,
        rejected_ids: Tensor,
        rejected_mask: Tensor,
        beta: float,
) -> Tensor:
    # (B,)
    chosen_logprobs: Tensor = get_sequence_logprobs(model, chosen_ids, chosen_mask)
    rejected_logprobs: Tensor = get_sequence_logprobs(model, rejected_ids, rejected_mask)

    with torch.no_grad():
        # (B,)
        ref_chosen_logprobs: Tensor = get_sequence_logprobs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_logprobs: Tensor = get_sequence_logprobs(ref_model, rejected_ids, rejected_mask)

    # DPO loss : (B,)
    logits = beta * (
            (chosen_logprobs - rejected_logprobs)
            - (ref_chosen_logprobs - ref_rejected_logprobs)
    )

    return -F.logsigmoid(logits).mean()  # scalar



# learning loop

tiny_stories_dpo_json = f"{SCRIPT_DIR}/dataset/storybot/tiny_stories_dpo.json"

gpt_model_storybot_dpo_pt = f"{SCRIPT_DIR}/.tmp/gpt_model_storybot_dpo.pt"

#context_len = 256
#batch_size = 8
#learning_rate = 5e-6
#beta = 0.1
#max_iters = 1000
#
#tokenizer = BPETokenizer.load_from(tiny_stories_merge_rules_pkl)
#dataset = DPODataset(tiny_stories_dpo_json, tokenizer, context_len)
#dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
#
#model = GPT.load_from(gpt_model_storybot_pretrain_pt, device = device)
#ref_model = GPT.load_from(gpt_model_storybot_pretrain_pt, device = device)
#ref_model.eval()
#
#optimizer = AdamW(model.parameters(), lr = learning_rate)
#
#losses = []
#data_iter = cycle(dataloader)
#pbar = tqdm(range(max_iters))
#
#for i in pbar:
#    chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)
#
#    chosen_ids = chosen_ids.to(device)
#    chosen_mask = chosen_mask.to(device)
#    rejected_ids = rejected_ids.to(device)
#    rejected_mask = rejected_mask.to(device)
#
#    loss = compute_dpo_loss(
#            model,
#            ref_model,
#            chosen_ids,
#            chosen_mask,
#            rejected_ids,
#            rejected_mask,
#            beta,
#    )
#
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#
#    losses.append(loss.item())
#    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
#
#plt.figure(figsize = (10, 6))
#plt.plot(losses)
#plt.xlabel("iteration")
#plt.ylabel("loss")
#plt.grid(True)
#plt.show()
#
#model.save_to(gpt_model_storybot_dpo_pt)



print(f"# 7 : Tokenizer Challenge")

owt_train_txt = f"{SCRIPT_DIR}/.tmp/owt_train.txt"
owt_valid_txt = f"{SCRIPT_DIR}/.tmp/owt_valid.txt"

webbot_merge_rules_pkl = f"{SCRIPT_DIR}/dataset/webbot/webbot_merge_rules.pkl"

if __name__ == "__main__":
    vocab_size = 50000
    merge_rules = train_bpe(owt_train_txt, vocab_size, num_processes = 2, num_chunks = 64)

    with open(webbot_merge_rules_pkl, "wb") as f:
        pickle.dump(merge_rules, f)

