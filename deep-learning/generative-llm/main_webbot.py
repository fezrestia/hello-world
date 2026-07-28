#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Any, Self, Iterator, cast, override, overload, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm  # type: ignore[import-untyped]
from collections import defaultdict
import re
import regex  # type: ignore[import-untyped]
from collections.abc import Iterable
import pickle
from multiprocessing import Pool
import shutil
from torch.optim.optimizer import Optimizer
from torch.amp import autocast
import wandb
import time
import random

import tokenizer_pyo3
rstk = cast(Any, tokenizer_pyo3)

SCRIPT_DIR = Path(__file__).resolve().parent
Path(f"{SCRIPT_DIR}/.tmp").mkdir(parents = True, exist_ok = True)

np.random.seed(0)
torch.manual_seed(0)

INT_INF = 1 << 60


tiny_codes_txt = f"{SCRIPT_DIR}/dataset/codebot/tiny_codes.txt"

tiny_stories_train_txt = f"{SCRIPT_DIR}/.tmp/tiny_stories_train.txt"
tiny_stories_valid_txt = f"{SCRIPT_DIR}/.tmp/tiny_stories_valid.txt"



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


def pretokenize(text: str) -> Iterator[str]:
    # regex in GPT-2
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    #return regex.findall(pattern, text)
    for m in regex.finditer(pattern, text):
        yield m.group(0)


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


def proc_pretoken_chunk(args: tuple[str, int, int, str]) -> dict[str, int]:
    file_path: str
    start: int
    end: int
    end_token: str
    file_path, start, end, end_token = args


    #pretoken_vs_count: dict[str, int] = rstk.proc_pretoken_chunk(
    #        file_path,
    #        start,
    #        end,
    #        end_token,
    #)


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
    print("python.train_bpe() : E")

    enter_ts = time.perf_counter()
    start_ts = time.perf_counter()

    chunk_boundaries: list[int] = find_chunk_boundaries(
            file_path,
            num_chunks = num_chunks,
            end_token = end_token,
    )
    total_chunks: int = len(chunk_boundaries) - 1

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    print(f"python.train_bpe() : find_chunk_boundaries() done [{elapsed_ms} ms]")
    start_ts = time.perf_counter()

    # paralell task
    chunk_info_list = []
    for chunk_idx in range(total_chunks):
        start: int = chunk_boundaries[chunk_idx]
        end: int = chunk_boundaries[chunk_idx + 1]
        chunk_info_list.append((file_path, start, end, end_token))

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    print(f"python.train_bpe() : parallel task def done [{elapsed_ms} ms]")
    start_ts = time.perf_counter()

    # run parallel
    with Pool(processes = num_processes) as pool:
        all_results: list[dict[str, int]] = list(tqdm(
                pool.imap(proc_pretoken_chunk, chunk_info_list),
                total = len(chunk_info_list),
                desc = "Pretokenizing",
        ))
    #all_results: list[dict[str, int]] = [proc_pretoken_chunk(chunk_info_list[0])]

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    print(f"python.train_bpe() : run parallel done [{elapsed_ms} ms]")
    start_ts = time.perf_counter()

    # merge result
    pretoken_vs_count: dict[str, int] = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_vs_count[pretoken] += count

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    print(f"python.train_bpe() : merge result done [{elapsed_ms} ms]")
    start_ts = time.perf_counter()

    # pretoken -> id
    #ids_vs_count: dict[tuple[int, ...], int] = {
    #        tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_vs_count.items()
    #}
    ids_vs_count: dict[tuple[int, ...], int] = rstk.encode_pretoken(pretoken_vs_count)

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    print(f"python.train_bpe() : pretoken -> id done [{elapsed_ms} ms]")
    start_ts = time.perf_counter()

    # 256 : default vocal size (1 byte)
    # 1 : end token
    num_merges: int = target_vocab_size - 256 - 1
    merge_rules: list[list[int]] = []  # [id_1, id_2, pair_id]

    # gen cache
    pair_vs_count: dict[tuple[int,int], int] = defaultdict(int)
    pair_vs_ids: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    #for ids, count in ids_vs_count.items():
    #    count_pairs(list(ids), count, pair_vs_count)
    #    for pair in zip(ids, ids[1:]):  # [0, 1, 2, 3] and [1, 2, 3] -> (0, 1), (1, 2), (2, 3)
    #        pair_vs_ids[pair].add(ids)  # register to cache
    (pair_vs_count, pair_vs_ids) = rstk.gen_cache(ids_vs_count)

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    print(f"python.train_bpe() : register cache done [{elapsed_ms} ms]")
    start_ts = time.perf_counter()

    merge_rules = rstk.train_bpe_loop(
            num_merges,
            pair_vs_count,
            pair_vs_ids,
            ids_vs_count,
    )

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    print(f"python.train_bpe() : train_bpe_loop done [{elapsed_ms} ms]")
    start_ts = time.perf_counter()

    # list -> dict
    merge_rules_dict: dict[tuple[int, int], int] = {}
    for (id_1, id_2, new_id) in merge_rules:
        merge_rules_dict[(id_1, id_2)] = new_id

    elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
    print(f"python.train_bpe() : list -> dict done [{elapsed_ms} ms]")
    start_ts = time.perf_counter()

    print(f"python.train_bpe() : X [{(time.perf_counter() - enter_ts) * 1000.0} ms]")
    return merge_rules_dict


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
            optimizer_state_dict: dict[str, Tensor]|None = None,
            iteration: int = 0,
    ) -> None:
        super().__init__()

        self.vocab_size: int = vocab_size
        self.max_context_len: int = max_context_len
        self.embed_dim: int = embed_dim
        self.head_count: int = head_count
        self.layer_count: int = layer_count
        self.ffn_hidden_dim: int = ffn_hidden_dim
        self.theta: int = theta

        # for train
        self.optimizer_state_dict: dict[str, Tensor]|None = optimizer_state_dict
        self.iteration: int = iteration

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

    def save_to(self, optimizer: Optimizer, file_path: str) -> None:
        checkpoint = {
                "model_state_dict": self.state_dict(),
                "vocab_size": self.vocab_size,
                "max_context_len": self.max_context_len,
                "embed_dim": self.embed_dim,
                "head_count": self.head_count,
                "layer_count": self.layer_count,
                "ffn_hidden_dim": self.ffn_hidden_dim,
                "theta": self.theta,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
                "np_random_state": np.random.get_state(),
                "py_random": random.getstate(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": self.iteration,
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
                optimizer_state_dict = checkpoint["optimizer_state_dict"],
                iteration = checkpoint["iteration"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["np_random_state"])
        random.setstate(checkpoint["py_random"])

        return model


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



print(f"# 7 : Tokenizer Challenge")

owt_train_txt = f"{SCRIPT_DIR}/.tmp/owt_train.txt"
owt_valid_txt = f"{SCRIPT_DIR}/.tmp/owt_valid.txt"

webbot_merge_rules_pkl = f"{SCRIPT_DIR}/dataset/webbot/webbot_merge_rules.pkl"

#if __name__ == "__main__":
#    #vocab_size = 1000
#    #merge_rules = train_bpe(
#    #        tiny_codes_txt,
#    #        vocab_size,
#    #        num_processes = 1,
#    #        num_chunks = 1,
#    #)
#
#    #vocab_size = 10000
#    #merge_rules = train_bpe(
#    #        tiny_stories_train_txt,
#    #        vocab_size,
#    #        num_processes = 2,
#    #        num_chunks = 256,
#    #)
#
#    vocab_size = 50000
#    merge_rules = train_bpe(
#            owt_train_txt,
#            vocab_size,
#            num_processes = 12,
#            num_chunks = 2048,
#    )
#
#    print(f"merge_rules len = {len(merge_rules)}")
#
#    with open(webbot_merge_rules_pkl, "wb") as f:
#        pickle.dump(merge_rules, f)


owt_train_bin = f"{SCRIPT_DIR}/.tmp/owt_train.bin"
owt_valid_bin = f"{SCRIPT_DIR}/.tmp/owt_valid.bin"

#if __name__ == '__main__':
#    tokenizer = BPETokenizer.load_from(webbot_merge_rules_pkl)
#
#    tokenizer.encode_file(
#            owt_train_txt,
#            owt_train_bin,
#            num_processes = 12,
#            num_chunks = 2048,
#    )
#
#    tokenizer.encode_file(
#            owt_valid_txt,
#            owt_valid_bin,
#            num_processes = 12,
#            num_chunks = 2048,
#    )



print(f"# 8 : Model Challenge")



print(f"# 9 : Learning Challenge")

gpt_model_webbot_pretrain_pt = f"{SCRIPT_DIR}/.tmp/gpt_model_webbot_pretrain.pt"

# hyper param

context_len = 1024
vocab_size = 50000
micro_batch_size = 32
accumulation_steps = 4
learning_rate = 6e-4  # max
warmup_iters = 0 #500
max_iters = 1000 #100000
embed_dim = 768
n_head = 16
n_layer = 12
ff_dim = 2048
theta = 10000
eval_iters = 100 #1000
grad_clip = 1.0

#if __name__ == '__main__':
#    wandb.init(project = "gpt-model-webbot-pretrain", config = {})
#    wandb.config.update({
#            "context_len": context_len,
#            "vocab_size": vocab_size,
#            "micro_batch_size": micro_batch_size,
#            "accumulation_steps": accumulation_steps,
#            "learning_rate": learning_rate,
#            "warmup_iters": warmup_iters,
#            "max_iters": max_iters,
#            "embed_dim": embed_dim,
#            "n_head": n_head,
#            "n_layer": n_layer,
#            "ff_dim": ff_dim,
#            "theta": theta,
#            "eval_iters": eval_iters,
#            "grad_clip": grad_clip,
#    })

# load data
train_data = np.memmap(owt_train_bin, dtype = np.uint16, mode = "r")
valid_data = np.memmap(owt_valid_bin, dtype = np.uint16, mode = "r")
print(f"owt_train.bin : {len(train_data)} tokens")
print(f"owt_valid.bin : {len(valid_data)} tokens")

# components
model_path = Path(gpt_model_webbot_pretrain_pt)
model: Any
if model_path.exists():
    print("Continue Training ...")
    model = GPT.load_from(gpt_model_webbot_pretrain_pt, device)
else:
    print("Train New Model")
    model = GPT(
            vocab_size,
            context_len,
            embed_dim,
            n_head,
            n_layer,
            ff_dim,
            theta,
    ).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"parameter count : {num_params} ({num_params/1e6:.2f}M)")

model = torch.compile(model)

optimizer = AdamW(model.parameters(), lr = learning_rate)
if model.optimizer_state_dict is not None:
    optimizer.load_state_dict(model.optimizer_state_dict)

pbar = tqdm(range(max_iters))

train_losses = []
train_iters = []

#val_loss: Any = "N/A"
#val_losses = []
#val_iters = []

for i in pbar:
    model.iteration += 1

    # update learning rate
    lr = get_learning_rate(i, warmup_iters, max_iters, learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad()

    # gradient accumulation
    for micro_step in range(accumulation_steps):
        (batch_x, batch_y) = get_batch(train_data, context_len, micro_batch_size, device)

        with autocast(device_type = device.type, dtype = torch.bfloat16):
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            loss = loss / accumulation_steps

        loss.backward()  # accumulate

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


    # log per step
    train_loss = loss.item() * accumulation_steps
    #wandb.log({"train/loss": train_loss, "train/lr": lr}, step = i)

    train_losses.append(train_loss)
    train_iters.append(i)

    # evaluate
    #if (i % eval_iters) == 0 or i == max_iters - 1:
    #    val_loss = evaluate(model, valid_data, context_len, micro_batch_size, device)
    #    val_losses.append(val_loss)
    #    val_iters.append(i)

        #wandb.log({"val/loss": val_loss}, step = i)

    #pbar.set_postfix({"train_loss": f"{train_loss}", "val_loss": f"{val_loss}"})
    pbar.set_postfix({"train_loss": f"{train_loss}"})


# evaluate
val_loss = evaluate(model, valid_data, context_len, micro_batch_size, device)
print(f"val_loss = {val_loss}, iteration = {model.iteration}")


model.save_to(optimizer, gpt_model_webbot_pretrain_pt)

plt.figure(figsize = (10, 6))
plt.plot(train_iters, train_losses, label = "train")
#plt.plot(val_iters, val_losses, label = "validate")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.savefig(f"{SCRIPT_DIR}/.tmp/storybot_pretrain_loss_val.png")
plt.show()

#wandb.finish()



# generate

#max_new_tokens = 100
#temperature = 0.5
#
#prompts = [
#    "In 1991, Linus Torvalds created",
#    "Monday, Tuesday, Wednesday,",
#    "Python was created by",
#    "Machine learning is defined as",
#    "The capital of Japan is",
#]
#
#tokenizer = BPETokenizer.load_from(webbot_merge_rules_pkl)
#
#model = GPT.load_from(gpt_model_webbot_pretrain_pt, device)
#
#print()
#for i, prompt in enumerate(prompts, 1):
#    print(f"{"-" * 64}")
#    print(f"prompt sample {i} = {prompt}")
#    print(f"{"-" * 16}")
#
#    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)
#    print(response)
#    print()

