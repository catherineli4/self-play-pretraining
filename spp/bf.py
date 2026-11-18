"""Python interface to the universal machine (App. E).

The fast path compiles `bf.c` with gcc (OpenMP) on first use; a pure-Python
interpreter with identical semantics is kept for tests and as a fallback.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokens import expand_macros

_HERE = Path(__file__).resolve().parent
_LIB = None


def _build_lib() -> ctypes.CDLL | None:
    global _LIB
    if _LIB is not None:
        return _LIB
    src = _HERE / "bf.c"
    build = _HERE / "_build"
    build.mkdir(exist_ok=True)
    digest = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
    so = build / f"libbf_{digest}.so"
    if not so.exists():
        cmd = ["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", "-o", str(so), str(src)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            return None
    lib = ctypes.CDLL(str(so))
    lib.bf_run_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.bf_run_batch.restype = None
    _LIB = lib
    return lib


@dataclass
class ExecResult:
    outputs: np.ndarray    # uint8 [n, T], zero-padded
    n_emitted: np.ndarray  # int32 [n]
    max_depth: np.ndarray  # int32 [n] maximum dynamic loop depth
    steps: np.ndarray      # int64 [n] instructions executed


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def run_program_py(body: bytes, tape_size: int, step_budget: int, T: int, seed: int):
    """Pure-Python reference interpreter (same semantics as bf.c)."""
    s = body.decode("latin-1")
    if "F" in s:
        s = s[: s.index("F")]
    code = expand_macros(s)
    n = len(code)
    jump = [-1] * n
    stack = []
    for i, c in enumerate(code):
        if c == "[":
            stack.append(i)
        elif c == "]" and stack:
            j = stack.pop()
            jump[i] = j
            jump[j] = i
    tape = bytearray(tape_size)
    out = bytearray(T)
    head = pc = emitted = depth = maxd = steps = 0
    rng = seed & 0xFFFFFFFFFFFFFFFF
    while pc < n and steps < step_budget and emitted < T:
        c = code[pc]
        steps += 1
        if c == ">":
            head = 0 if head + 1 == tape_size else head + 1
        elif c == "<":
            head = tape_size - 1 if head == 0 else head - 1
        elif c == "+":
            tape[head] = (tape[head] + 1) & 0xFF
        elif c == "-":
            tape[head] = (tape[head] - 1) & 0xFF
        elif c == ".":
            out[emitted] = tape[head]
            emitted += 1
        elif c == ",":
            rng, z = _splitmix64(rng)
            tape[head] = z >> 56
        elif c == "[":
            if jump[pc] >= 0:
                if tape[head] == 0:
                    pc = jump[pc]
                else:
                    depth += 1
                    maxd = max(maxd, depth)
        elif c == "]":
            if jump[pc] >= 0:
                if tape[head] != 0:
                    pc = jump[pc]
                else:
                    depth -= 1
        pc += 1
    return bytes(out), emitted, maxd, steps


def _per_program_seed(seed: int, i: int) -> int:
    return (seed ^ ((0x9E3779B97F4A7C15 * (i + 1)) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF


def run_programs(bodies: list[bytes], *, tape_size: int = 1024, step_budget: int = 65536,
                 T: int = 4095, seed: int = 0, force_python: bool = False) -> ExecResult:
    """Execute a batch of program bodies (bytes without S/F) with fresh random tapes."""
    n = len(bodies)
    out = np.zeros((n, T), dtype=np.uint8)
    n_emitted = np.zeros(n, dtype=np.int32)
    max_depth = np.zeros(n, dtype=np.int32)
    steps = np.zeros(n, dtype=np.int64)
    if n == 0:
        return ExecResult(out, n_emitted, max_depth, steps)
    seed &= 0xFFFFFFFFFFFFFFFF
    lib = None if force_python else _build_lib()
    if lib is None:
        for i, b in enumerate(bodies):
            o, e, d, s = run_program_py(b, tape_size, step_budget, T, _per_program_seed(seed, i))
            out[i] = np.frombuffer(o, dtype=np.uint8)
            n_emitted[i], max_depth[i], steps[i] = e, d, s
        return ExecResult(out, n_emitted, max_depth, steps)
    lengths = np.array([len(b) for b in bodies], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    buf = np.frombuffer(b"".join(bodies), dtype=np.uint8).copy() if offsets[-1] > 0 else np.zeros(1, np.uint8)
    lib.bf_run_batch(
        buf.ctypes.data, offsets.ctypes.data, n, tape_size, step_budget, T, seed,
        out.ctypes.data, n_emitted.ctypes.data, max_depth.ctypes.data, steps.ctypes.data,
    )
    return ExecResult(out, n_emitted, max_depth, steps)


def have_fast_interpreter() -> bool:
    return _build_lib() is not None
