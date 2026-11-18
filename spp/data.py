"""Evaluation datasets (byte records), synthetic evals, and the non-adaptive baselines
(uniform program prior, random PCFG; paper Sec. 3.1 and App. H)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .model import per_sequence_cross_entropy
from .tokens import ALPHABET_SIZE, END_ID, PROGRAM_TOKEN_IDS, START_OUTPUT

LN2 = math.log(2.0)


# ---------------------------------------------------------------- byte records

def records_from_bytes(data: np.ndarray | bytes, record_len: int, max_records: int | None = None) -> np.ndarray:
    """Split a byte stream into non-overlapping records of `record_len` bytes (tail discarded)."""
    arr = np.frombuffer(bytes(data), dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else np.asarray(data, np.uint8)
    n = len(arr) // record_len
    if max_records is not None:
        n = min(n, max_records)
    return arr[: n * record_len].reshape(n, record_len)


def load_byte_file(path: str | Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint8)


def random_bytes_records(n: int, record_len: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, record_len), dtype=np.uint8)


def arithmetic_records(n: int, record_len: int, seed: int = 0) -> np.ndarray:
    """Arithmetic progressions mod 256 with random start and step (a sanity-check eval)."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, 256, size=(n, 1))
    step = rng.integers(1, 256, size=(n, 1))
    return ((start + step * np.arange(record_len)[None]) % 256).astype(np.uint8)


def to_learner_batch(records: np.ndarray, device) -> torch.Tensor:
    """Prefix each record with the O byte -> [n, record_len + 1] int64."""
    x = torch.from_numpy(np.ascontiguousarray(records)).long()
    prefix = torch.full((x.shape[0], 1), START_OUTPUT, dtype=torch.long)
    return torch.cat([prefix, x], 1).to(device)


@torch.no_grad()
def evaluate_bpb(model, records: np.ndarray, batch_size: int = 64, device="cuda") -> float:
    """Zero-shot next-byte loss in bits per byte over the content bytes of each record."""
    model.eval()
    total, count = 0.0, 0
    for s in range(0, len(records), batch_size):
        x = to_learner_batch(records[s:s + batch_size], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            logits = model(x[:, :-1])
        loss = per_sequence_cross_entropy(logits, x[:, 1:])
        total += float(loss.sum())
        count += len(loss)
    model.train()
    return total / max(count, 1) / LN2


# ---------------------------------------------------------------- baselines

def sample_uniform_programs(n: int, max_len: int, rng: np.random.Generator) -> tuple[list[bytes], np.ndarray]:
    """Solomonoff-style prior: i.i.d. uniform tokens over the 19-symbol alphabet until F."""
    toks = rng.integers(0, ALPHABET_SIZE, size=(n, max_len))
    ids = np.array(PROGRAM_TOKEN_IDS, dtype=np.uint8)[toks]
    bodies, logp = [], np.zeros(n)
    for i in range(n):
        hits = np.flatnonzero(ids[i] == END_ID)
        L = int(hits[0]) if len(hits) else max_len
        bodies.append(bytes(ids[i, :L].tolist()))
        logp[i] = -(L + (1 if L < max_len else 0)) * math.log(ALPHABET_SIZE)
    return bodies, logp


class RandomPCFG:
    """Random probabilistic context-free grammars and row packing (paper App. H)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        n_t = int(rng.integers(2, 17))
        self.terminals = rng.choice(np.arange(1, 256), size=n_t, replace=False).tolist()
        n_v = int(rng.integers(1, 9))
        self.rules: list[list[tuple[float, list[tuple[bool, int]]]]] = []
        for _ in range(n_v):
            n_p = int(rng.integers(1, 5))
            w = rng.uniform(0, 1, size=n_p) + 1e-6
            w = w / w.sum()
            prods = []
            for j in range(n_p):
                rhs = [self._symbol(n_v) for _ in range(int(rng.integers(1, 5)))]
                prods.append((float(w[j]), rhs))
            if not any(all(t for t, _ in rhs) for _, rhs in prods):  # productivity repair
                j = int(rng.integers(0, n_p))
                rhs = [(True, int(rng.choice(self.terminals))) for _ in range(int(rng.integers(1, 5)))]
                prods[j] = (prods[j][0], rhs)
            self.rules.append(prods)
        self.start_yield = next(rhs for _, rhs in self.rules[0] if all(t for t, _ in rhs))

    def _symbol(self, n_v):
        if self.rng.random() < 0.5:
            return (True, int(self.rng.choice(self.terminals)))
        return (False, int(self.rng.integers(0, n_v)))

    def derive_word(self, max_bytes: int = 64, max_expansions: int = 10_000) -> list[int]:
        out, stack, n_exp = [], [(False, 0)], 0
        while stack and len(out) < max_bytes and n_exp < max_expansions:
            is_t, s = stack.pop()
            if is_t:
                out.append(s)
                continue
            n_exp += 1
            prods = self.rules[s]
            j = self.rng.choice(len(prods), p=[p for p, _ in prods])
            stack.extend(reversed(prods[j][1]))
        if not out:
            out = [t for _, t in self.start_yield][:max_bytes]
        return out[:max_bytes]

    def row(self, row_len: int) -> np.ndarray:
        out: list[int] = []
        while len(out) < row_len:
            out.extend(self.derive_word())
        return np.array(out[:row_len], dtype=np.uint8)


def sample_pcfg_rows(n: int, row_len: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([RandomPCFG(rng).row(row_len) for _ in range(n)])
