"""Program pool machinery (paper App. G): replay bank, MAP-Elites archive, mutations."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .tokens import BODY_ALPHABET

LEN_BUCKETS = (8, 16, 32)   # body length buckets -> 4 bins
MAX_DEPTH_BIN = 8           # loop depth 0..8 (clamped) -> 9 bins; 36 niches total


@dataclass
class Program:
    body: bytes
    logp_old: float = 0.0   # log g_phi(x) under the generator that produced/evaluated it
    round: int = 0
    source: str = "fresh"   # fresh | mut | replay


def length_bin(n: int) -> int:
    for i, b in enumerate(LEN_BUCKETS):
        if n <= b:
            return i
    return len(LEN_BUCKETS)


def niche_of(body_len: int, max_depth: int) -> tuple[int, int]:
    return (min(max_depth, MAX_DEPTH_BIN), length_bin(body_len))


class ReplayBank:
    """Bank of non-replay programs from earlier rounds, sampled uniformly without replacement."""

    def __init__(self, capacity: int = 1_000_000, seed: int = 0):
        self.capacity = capacity
        self.items: list[Program] = []
        self.rng = random.Random(seed)
        self._write = 0

    def add(self, programs: list[Program]) -> None:
        for p in programs:
            if len(self.items) < self.capacity:
                self.items.append(p)
            else:  # ring-buffer overwrite of the oldest entries
                self.items[self._write] = p
                self._write = (self._write + 1) % self.capacity

    def sample(self, n: int) -> list[Program]:
        n = min(n, len(self.items))
        if n == 0:
            return []
        picks = self.rng.sample(self.items, n)
        return [Program(p.body, p.logp_old, p.round, "replay") for p in picks]

    def __len__(self):
        return len(self.items)

    def state_dict(self):
        return {"capacity": self.capacity, "items": [(p.body, p.logp_old, p.round, p.source) for p in self.items],
                "write": self._write, "rng": self.rng.getstate()}

    def load_state_dict(self, s):
        self.capacity = s["capacity"]
        self.items = [Program(*t) for t in s["items"]]
        self._write = s["write"]
        self.rng.setstate(s["rng"])


@dataclass
class Elite:
    body: bytes
    reward: float
    round: int = 0


@dataclass
class EliteArchive:
    """MAP-Elites archive keyed by (max dynamic loop depth, body length bucket)."""
    per_niche: int = 8
    decay: float = 0.97
    seed: int = 0
    niches: dict[tuple[int, int], list[Elite]] = field(default_factory=dict)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def step_decay(self) -> None:
        for elites in self.niches.values():
            for e in elites:
                e.reward *= self.decay

    def add(self, body: bytes, reward: float, max_depth: int, round: int = 0) -> bool:
        if reward <= 0:
            return False
        key = niche_of(len(body), max_depth)
        elites = self.niches.setdefault(key, [])
        for e in elites:  # keep one copy of each program per niche
            if e.body == body:
                if reward > e.reward:
                    e.reward, e.round = reward, round
                elites.sort(key=lambda z: -z.reward)
                return True
        elites.append(Elite(body, reward, round))
        elites.sort(key=lambda z: -z.reward)
        del elites[self.per_niche:]
        return True

    def sample_parent(self) -> Elite | None:
        keys = [k for k, v in self.niches.items() if v]
        if not keys:
            return None
        key = self.rng.choice(keys)            # uniform over occupied niches
        return self.rng.choice(self.niches[key])

    def mutate(self, body: bytes, max_len: int) -> bytes:
        """Single-token substitution, insertion or deletion of the program body."""
        b = bytearray(body)
        ops = ["sub", "ins", "del"] if len(b) > 1 else (["sub", "ins"] if len(b) == 1 else ["ins"])
        if len(b) >= max_len and "ins" in ops:
            ops.remove("ins")
        op = self.rng.choice(ops)
        tok = ord(self.rng.choice(BODY_ALPHABET))
        if op == "sub":
            b[self.rng.randrange(len(b))] = tok
        elif op == "ins":
            b.insert(self.rng.randint(0, len(b)), tok)
        else:
            del b[self.rng.randrange(len(b))]
        return bytes(b)

    def propose(self, n: int, max_len: int) -> list[bytes]:
        out = []
        for _ in range(n):
            parent = self.sample_parent()
            if parent is None:
                break
            out.append(self.mutate(parent.body, max_len))
        return out

    @property
    def n_occupied(self) -> int:
        return sum(1 for v in self.niches.values() if v)

    @property
    def n_programs(self) -> int:
        return sum(len(v) for v in self.niches.values())

    def state_dict(self):
        return {"per_niche": self.per_niche, "decay": self.decay, "rng": self.rng.getstate(),
                "niches": {k: [(e.body, e.reward, e.round) for e in v] for k, v in self.niches.items()}}

    def load_state_dict(self, s):
        self.per_niche, self.decay = s["per_niche"], s["decay"]
        self.rng.setstate(s["rng"])
        self.niches = {k: [Elite(*t) for t in v] for k, v in s["niches"].items()}
