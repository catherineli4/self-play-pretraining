"""Byte-level re-implementations of the synthetic pre-pretraining curricula compared in "Rethinking
Pre-pretraining" (Stern, Rocktaschel, Locatelli, Ruis, 2026; arXiv:2610.40051):

  * k-Dyck (Hu et al., 2025): balanced brackets with k = 128 types and nesting depth <= 16. Bytes 0..127 open
    bracket type t, bytes 128..255 close it. The stream is a concatenation of balanced 4094-byte words.
  * Neural cellular automata (Lee et al., 2026): 12x12 grids of 10 colours evolved by a randomly initialised
    CNN (per-trajectory random weights, circular padding, damped tanh update), serialised frame by frame
    (144 colour bytes 0..9 + separator 10; trajectories separated by 11). Trajectories are kept only if gzip
    compresses them by at least 60% (removes noise-like rules) and by at most 92% (removes frozen ones).
  * DeclRef ("declarative reference corpora", the paper's own curriculum): nested scopes in which identifiers
    are declared and later referenced. Bytes: identifiers 0..199, OPEN 250, CLOSE 251, DECL 252, REF 253,
    statement separator 254, end padding 255. References pick visible identifiers with a recency bias.

The originals are token-level and their exact generators were not available; the grammars follow the
papers' descriptions. All generators are vectorised over many sequences and return uint8 arrays.
"""
from __future__ import annotations

import gzip

import numpy as np

RECORD_LEN = 4095


def kdyck_batch(n_seq: int, length: int, rng: np.random.Generator, k: int = 128, max_depth: int = 16,
                p_open: float = 0.5) -> np.ndarray:
    stack = np.zeros((n_seq, max_depth), np.uint8)
    depth = np.zeros(n_seq, np.int64)
    out = np.empty((n_seq, length), np.uint8)
    ar = np.arange(n_seq)
    for i in range(length):
        remaining = length - i
        must_close = (depth >= remaining) | (depth >= max_depth)
        open_ = np.where(depth == 0, True, np.where(must_close, False, rng.random(n_seq) < p_open))
        t = rng.integers(0, k, n_seq).astype(np.uint8)
        o = ar[open_]
        stack[o, depth[o]] = t[o]
        out[o, i] = t[o]
        depth[o] += 1
        c = ar[~open_]
        depth[c] -= 1
        out[c, i] = 128 + stack[c, depth[c]]
    return out


def kdyck_stream(n_bytes: int, seed: int = 0, batch: int = 8192, word_len: int = RECORD_LEN - 1, **kw) -> np.ndarray:
    """Concatenated balanced words of even length (an odd length would leave one bracket open)."""
    rng = np.random.default_rng(seed)
    chunks, total = [], 0
    while total < n_bytes:
        b = kdyck_batch(batch, word_len - word_len % 2, rng, **kw).reshape(-1)
        chunks.append(b)
        total += len(b)
    return np.concatenate(chunks)[:n_bytes]


def nca_batch(n_traj: int, steps: int, gen, device, grid: int = 12, colors: int = 10, channels: int = 16,
              gain: tuple[float, float] = (0.5, 2.0), step_size: float = 0.3,
              keep: tuple[float, float] = (0.08, 0.4)) -> list[bytes]:
    """One random 3x3 CNN rule per trajectory (weight gain log-uniform in `gain`), applied as a damped
    update s <- (1 - step_size) s + step_size tanh(conv(s) + b) so that dynamics persist instead of
    freezing within a few steps; colours are the argmax over the first `colors` channels."""
    import math
    import torch
    import torch.nn.functional as F
    B, C, G = n_traj, channels, grid
    lo, hi = math.log(gain[0]), math.log(gain[1])
    g = torch.exp(torch.rand(B, generator=gen, device=device) * (hi - lo) + lo) / (9 * C) ** 0.5
    w = (torch.randn(B, C, C, 3, 3, generator=gen, device=device) * g[:, None, None, None, None]).reshape(B * C, C, 3, 3)
    bias = torch.randn(B, C, 1, 1, generator=gen, device=device) * 0.1
    s = torch.rand(B, C, G, G, generator=gen, device=device) * 2 - 1
    frames = []
    for _ in range(steps):
        x = F.pad(s.reshape(1, B * C, G, G), (1, 1, 1, 1), mode="circular")
        upd = torch.tanh(F.conv2d(x, w, groups=B).reshape(B, C, G, G) + bias)
        s = (1 - step_size) * s + step_size * upd
        frames.append(s[:, :colors].argmax(1).to(torch.uint8))          # (B, G, G)
    fr = torch.stack(frames, 1).reshape(B, steps, G * G).cpu().numpy()  # (B, steps, 144)
    sep = np.full((B, steps, 1), 10, np.uint8)
    traj = np.concatenate([fr, sep], 2).reshape(B, -1)
    out = []
    for t in traj:
        b = t.tobytes()
        ratio = len(gzip.compress(b, 6)) / len(b)
        if keep[0] < ratio < keep[1]:
            out.append(b + bytes([11]))
    return out


def nca_stream(n_bytes: int, seed: int = 0, batch: int = 1024, steps: int = 28, device: str | None = None,
               verbose: bool = False, **kw) -> np.ndarray:
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device).manual_seed(seed)
    chunks, total, tried, kept = [], 0, 0, 0
    while total < n_bytes:
        trajs = nca_batch(batch, steps, gen, device, **kw)
        tried += batch
        kept += len(trajs)
        for b in trajs:
            chunks.append(np.frombuffer(b, np.uint8))
            total += len(b)
    if verbose:
        print(f"nca: kept {kept}/{tried} trajectories ({100*kept/max(tried,1):.0f}%)", flush=True)
    return np.concatenate(chunks)[:n_bytes]


OPEN, CLOSE, DECL, REF, SEP, PAD = 250, 251, 252, 253, 254, 255


def declref_batch(n_seq: int, length: int, rng: np.random.Generator, n_ids: int = 200, max_depth: int = 8,
                  cap: int = 48, probs=(0.15, 0.15, 0.30, 0.40), recency: float = 0.5) -> np.ndarray:
    N = n_seq
    vis = np.zeros((N, cap), np.uint8)
    nvis = np.zeros(N, np.int64)
    depth = np.zeros(N, np.int64)
    scope = np.zeros((N, max_depth), np.int64)
    out = np.full((N, length + 4), SEP, np.uint8)
    cur = np.zeros(N, np.int64)
    ar = np.arange(N)
    p = np.asarray(probs) / np.sum(probs)
    while True:
        active = cur <= length - 3
        if not active.any():
            break
        a = rng.choice(4, size=N, p=p)
        a[(a == 1) & (depth == 0)] = 2
        a[(a == 3) & (nvis == 0)] = 2
        a[(a == 0) & (depth >= max_depth)] = 3
        a[(a == 3) & (nvis == 0)] = 2
        full = (a == 2) & (nvis >= cap)
        a[full] = np.where(depth[full] > 0, 1, 3)
        a[~active] = -1
        # OPEN
        i = ar[a == 0]
        scope[i, depth[i]] = nvis[i]
        depth[i] += 1
        out[i, cur[i]] = OPEN
        out[i, cur[i] + 1] = SEP
        cur[i] += 2
        # CLOSE
        i = ar[a == 1]
        depth[i] -= 1
        nvis[i] = scope[i, depth[i]]
        out[i, cur[i]] = CLOSE
        out[i, cur[i] + 1] = SEP
        cur[i] += 2
        # DECL: a fresh identifier not currently visible
        i = ar[a == 2]
        ident = rng.integers(0, n_ids, len(i)).astype(np.uint8)
        for _ in range(8):
            visible = (vis[i] == ident[:, None]) & (np.arange(cap)[None, :] < nvis[i, None])
            clash = visible.any(1)
            if not clash.any():
                break
            ident[clash] = rng.integers(0, n_ids, int(clash.sum())).astype(np.uint8)
        vis[i, nvis[i]] = ident
        nvis[i] += 1
        out[i, cur[i]] = DECL
        out[i, cur[i] + 1] = ident
        out[i, cur[i] + 2] = SEP
        cur[i] += 3
        # REF: a visible identifier, recent ones preferred
        i = ar[a == 3]
        back = rng.geometric(recency, len(i)) - 1
        pos = np.clip(nvis[i] - 1 - back, 0, None)
        out[i, cur[i]] = REF
        out[i, cur[i] + 1] = vis[i, pos]
        out[i, cur[i] + 2] = SEP
        cur[i] += 3
    out[:, length:] = PAD
    res = out[:, :length].copy()
    for j in range(N):                       # pad the unused tail of each record
        res[j, cur[j]:] = PAD
    return res


def declref_stream(n_bytes: int, seed: int = 0, batch: int = 8192, **kw) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks, total = [], 0
    while total < n_bytes:
        b = declref_batch(batch, RECORD_LEN, rng, **kw).reshape(-1)
        chunks.append(b)
        total += len(b)
    return np.concatenate(chunks)[:n_bytes]


GENERATORS = {"kdyck": kdyck_stream, "nca": nca_stream, "declref": declref_stream}
