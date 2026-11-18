"""Scale-parametrised Gaussian initialization for the from-scratch arm of the
pre-pretraining experiment (paper App. A; scale sensitivity per arXiv:2610.40051).

All linear weights ~ N(0, std); token embedding ~ N(0, emb_std); LM head ~ N(0, head_std)
(0 = zero head); the residual output projections (attention wo, MLP w2) are multiplied by
resid_scale, given either as a number or as the string "1/sqrt(2L)". RMSNorm gains stay 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from .model import Llama


@dataclass
class InitScales:
    std: float = 0.02               # all linear layers inside the blocks
    emb_std: float = 0.02           # token embedding
    head_std: float = 0.02          # LM head (0 -> zero-initialised head)
    resid_scale: str | float = "1/sqrt(2L)"   # multiplier on wo / w2 after drawing N(0, std)
    seed: int = 0

    def resid_factor(self, n_layers: int) -> float:
        if isinstance(self.resid_scale, str):
            s = self.resid_scale.replace(" ", "")
            if s in ("1/sqrt(2L)", "scaled"):
                return 1.0 / math.sqrt(2 * n_layers)
            return float(s)
        return float(self.resid_scale)

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def apply_init(model: Llama, scales: InitScales) -> dict:
    """Re-initialise `model` in place; returns a summary of the per-group stds actually used."""
    g = torch.Generator(device="cpu").manual_seed(scales.seed)
    L = model.cfg.n_layers
    rf = scales.resid_factor(L)

    def normal_(w: torch.Tensor, std: float):
        if std <= 0:
            w.zero_()
        else:
            w.copy_(torch.randn(w.shape, generator=g) * std)

    summary = {"std": scales.std, "emb_std": scales.emb_std, "head_std": scales.head_std,
               "resid_factor": rf, "resid_std": scales.std * rf}
    normal_(model.tok_emb.weight, scales.emb_std)
    normal_(model.lm_head.weight, scales.head_std)
    for blk in model.blocks:
        for lin in (blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.mlp.w1, blk.mlp.w3):
            normal_(lin.weight, scales.std)
        for lin in (blk.attn.wo, blk.mlp.w2):
            normal_(lin.weight, scales.std * rf)
        blk.attn_norm.weight.fill_(1.0)
        blk.mlp_norm.weight.fill_(1.0)
    model.norm.weight.fill_(1.0)
    return summary
