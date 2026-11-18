"""In-context learning tasks and evaluation (paper Sec. 3.2, App. E "ICL Tasks").

Each example is [0, x_1..x_k, f(x_1..x_k)]; after m examples the query's sentinel
and inputs are given and the model must produce f(query) under greedy decoding.
"""
from __future__ import annotations

import numpy as np
import torch

from .tokens import START_OUTPUT

PUSH, POP = 250, 251
TASKS = ("reverse", "stack", "assoc_recall", "sum", "max", "min")


def _nz(rng, size, high=256):
    return rng.integers(1, high, size=size)


def make_example(task: str, rng: np.random.Generator, k: int = 4, dictionary=None):
    """Return (inputs, answer) as lists of bytes (without the sentinel)."""
    if task == "reverse":
        x = _nz(rng, k).tolist()
        return x, x[::-1]
    if task == "sum":
        a, b = int(_nz(rng, 1)[0]), int(_nz(rng, 1)[0])
        return [a, b], [(a + b) % 256]
    if task == "max":
        x = _nz(rng, k).tolist()
        return x, [max(x)]
    if task == "min":
        x = _nz(rng, k).tolist()
        return x, [min(x)]
    if task == "assoc_recall":
        key = int(rng.choice(list(dictionary.keys())))
        return [key], [dictionary[key]]
    if task == "stack":
        ops, stack = [], []
        n_ops = int(rng.integers(2, 2 * k + 1))
        for _ in range(n_ops):
            if stack and rng.random() < 0.5:
                ops += [POP, stack.pop()]
            else:
                v = int(rng.integers(1, PUSH))
                stack.append(v)
                ops += [PUSH, v]
        if not stack:
            v = int(rng.integers(1, PUSH))
            stack.append(v)
            ops += [PUSH, v]
        return ops + [POP], [stack.pop()]
    raise ValueError(task)


def make_instance(task: str, m: int, rng: np.random.Generator, k: int = 4, dict_size: int = 8):
    """Build one ICL sequence with m demonstrations. Returns (prefix tokens, answer tokens)."""
    dictionary = None
    seq = []
    if task == "assoc_recall":
        keys = rng.choice(np.arange(1, 256), size=dict_size, replace=False)
        dictionary = {int(kk): int(v) for kk, v in zip(keys, _nz(rng, dict_size))}
        for kk, v in dictionary.items():  # print the dictionary first
            seq += [0, kk, v]
    for _ in range(m):
        x, y = make_example(task, rng, k, dictionary)
        seq += [0, *x, *y]
    x, y = make_example(task, rng, k, dictionary)
    seq += [0, *x]
    return seq, y


@torch.no_grad()
def evaluate_icl(model, tasks=TASKS, m_values=(0, 1, 2, 4, 8, 16, 32), n_instances: int = 128,
                 k: int = 4, ctx: int = 4096, device="cuda", seed: int = 0, batch_size: int = 64):
    """Greedy success rate and mean probability of the correct answer per task and m."""
    model.eval()
    results: dict[str, dict[int, dict[str, float]]] = {}
    for task in tasks:
        results[task] = {}
        for m in m_values:
            rng = np.random.default_rng(seed + 1000 * m)
            seqs, answers = [], []
            for _ in range(n_instances):
                p, a = make_instance(task, m, rng, k)
                full = [START_OUTPUT, *p, *a]
                if len(full) > ctx:
                    continue
                seqs.append(full)
                answers.append(len(a))
            if not seqs:
                continue
            succ, probs = [], []
            for s in range(0, len(seqs), batch_size):
                chunk = seqs[s:s + batch_size]
                L = max(len(q) for q in chunk)
                x = torch.zeros((len(chunk), L), dtype=torch.long)
                for i, q in enumerate(chunk):
                    x[i, : len(q)] = torch.tensor(q)
                x = x.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                    logits = model(x)
                lp = torch.log_softmax(logits.float(), -1)
                for i, q in enumerate(chunk):
                    n_ans = answers[s + i]
                    pos = torch.arange(len(q) - n_ans - 1, len(q) - 1, device=device)
                    tgt = x[i, pos + 1]
                    pred = lp[i, pos].argmax(-1)
                    succ.append(float((pred == tgt).all()))
                    probs.append(float(lp[i, pos].gather(-1, tgt[:, None]).sum().exp()))
            results[task][m] = {"success": float(np.mean(succ)), "p_correct": float(np.mean(probs)),
                                "n": len(succ)}
    model.train()
    return results
