"""Detection of mathematical sequence families in program outputs (paper App. D).

Families mod 256: arithmetic / quadratic / cubic (constant 1st/2nd/3rd finite
differences, lowest order wins), Fibonacci-like (v_n = v_{n-1} + v_{n-2}), and
geometric (v_n = r v_{n-1}). Up to 30 unrelated leading bytes are allowed and the
matched region must have minimal period >= 30 (full region and trailing window).
"""
from __future__ import annotations

import numpy as np

MAX_LEAD = 30
MIN_PERIOD = 30
MIN_LEN = 32
TRAIL = 64
FAMILIES = ("arithmetic", "quadratic", "cubic", "fibonacci", "geometric")


def has_period_below(v: np.ndarray, bound: int) -> bool:
    """True if v has a period p < bound (so its minimal period is < bound)."""
    for p in range(1, min(bound, len(v))):
        if np.array_equal(v[:-p], v[p:]):
            return True
    return False


def _nondegenerate(v: np.ndarray) -> bool:
    if len(v) < MIN_LEN:
        return False
    return not has_period_below(v, MIN_PERIOD) and not has_period_below(v[-TRAIL:], MIN_PERIOD)


def _is_const(d: np.ndarray) -> bool:
    return len(d) > 0 and bool(np.all(d == d[0]))


def _family(v: np.ndarray) -> str | None:
    v = v.astype(np.int64) % 256
    d1 = np.diff(v) % 256
    if _is_const(d1):
        return "arithmetic"
    d2 = np.diff(d1) % 256
    if _is_const(d2):
        return "quadratic"
    d3 = np.diff(d2) % 256
    if _is_const(d3):
        return "cubic"
    if np.all((v[2:] - v[1:-1] - v[:-2]) % 256 == 0):
        return "fibonacci"
    odd = np.flatnonzero(v[:-1] % 2 == 1)
    if len(odd):  # ratio is determined by any invertible predecessor
        i = int(odd[0])
        r = (int(v[i + 1]) * pow(int(v[i]), -1, 256)) % 256
        if np.all((v[1:] - r * v[:-1]) % 256 == 0):
            return "geometric"
    else:
        for r in range(256):
            if np.all((v[1:] - r * v[:-1]) % 256 == 0):
                return "geometric"
    return None


def classify_output(y, n_emitted: int | None = None) -> str | None:
    """Return the family name of the structured suffix of an output tape, or None."""
    v = np.frombuffer(bytes(y), dtype=np.uint8) if isinstance(y, (bytes, bytearray)) else np.asarray(y, dtype=np.uint8)
    if n_emitted is not None:
        v = v[:n_emitted]
    for lead in range(0, min(MAX_LEAD, max(len(v) - MIN_LEN, 0)) + 1):
        region = v[lead:]
        if len(region) < MIN_LEN:
            break
        fam = _family(region)
        if fam is not None and _nondegenerate(region):
            return fam
    return None
