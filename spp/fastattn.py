"""Attention dispatch: flash SDPA for ordinary forward/backward, and (optionally) the
JVP flash-attention Triton kernel of https://github.com/amorehead/jvp_flash_attention
for the forward-mode reward. Without that package the reward falls back to PyTorch's
math SDPA backend, the only built-in backend that supports forward-mode AD.

    pip install jvp_flash_attention      # torch>=2.8; or set SPP_PYLIBS=/dir/with/the/package
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

_extra = os.environ.get("SPP_PYLIBS")
if _extra and _extra not in sys.path:
    sys.path.insert(0, _extra)

try:  # optional dependency
    from jvp_flash_attention.jvp_attention import JVPAttn as _JVPAttn
    JVP_FLASH_AVAILABLE = True
except Exception:  # noqa: BLE001
    _JVPAttn = None
    JVP_FLASH_AVAILABLE = False

_MODE = "default"   # default | jvp_flash


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal attention on [B, H, T, D] tensors."""
    if _MODE == "jvp_flash":
        return _JVPAttn.fwd_dual(q, k, v, causal=True)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


@contextmanager
def forward_mode_attention(kernel: str = "auto"):
    """Context for torch.func.jvp: kernel = auto | flash | math."""
    global _MODE
    use_flash = (kernel == "flash") or (kernel == "auto" and JVP_FLASH_AVAILABLE)
    if kernel == "flash" and not JVP_FLASH_AVAILABLE:
        raise RuntimeError("jvp_kernel=flash but jvp_flash_attention is not importable")
    if use_flash:
        _MODE = "jvp_flash"
        try:
            yield "flash"
        finally:
            _MODE = "default"
    else:
        with sdpa_kernel(SDPBackend.MATH):
            yield "math"
