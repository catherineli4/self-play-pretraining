"""Llama-style decoder-only transformer with a 256-byte vocabulary (paper Sec. 2.3).

RMSNorm, rotary position embeddings, SwiGLU MLP, no biases, untied output head.
The forward is written so that `torch.func.functional_call` + `torch.func.jvp`
work through it (needed for the forward-mode reward in `reward.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fastattn
from .tokens import VOCAB_SIZE


@dataclass
class ModelConfig:
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 352
    max_seq_len: int = 4096
    vocab_size: int = VOCAB_SIZE
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def to_dict(self):
        return asdict(self)


# Parameter counts (incl. embeddings): tiny 0.13M, 1m 0.87M, 3m 2.8M, 8m 6.6M, 25m 24.5M. Head dim is 32 or 64
# everywhere (the JVP flash kernel supports {16, 32, 64, 128, 256}).
PRESETS = {
    "tiny": ModelConfig(dim=64, n_layers=2, n_heads=2, ffn_dim=176),
    "1m": ModelConfig(dim=128, n_layers=4, n_heads=4, ffn_dim=352),
    "3m": ModelConfig(dim=192, n_layers=6, n_heads=6, ffn_dim=512),
    "8m": ModelConfig(dim=256, n_layers=8, n_heads=8, ffn_dim=704),
    "25m": ModelConfig(dim=512, n_layers=8, n_heads=8, ffn_dim=1280),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def _rope_cache(max_seq_len: int, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv)  # [T, D/2]
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x, cos, sin):
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, : x.shape[2]].to(x.dtype)
    s = sin[None, None, : x.shape[2]].to(x.dtype)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = fastattn.attention(q, k, v)
        return self.wo(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Llama(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        cos, sin = _rope_cache(cfg.max_seq_len, cfg.dim // cfg.n_heads, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [B, T] int64 -> logits [B, T, vocab] (float32)."""
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        return self.lm_head(self.norm(x)).float()

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.lm_head.weight.numel()
        return n


def per_sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                               mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean next-token cross-entropy per sequence (nats). logits [B,T,V], targets [B,T]."""
    lp = F.log_softmax(logits.float(), dim=-1)
    nll = -lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T]
    if mask is None:
        return nll.mean(-1)
    m = mask.to(nll.dtype)
    return (nll * m).sum(-1) / m.sum(-1).clamp_min(1.0)


def build_model(preset_or_cfg: str | ModelConfig, max_seq_len: int | None = None) -> Llama:
    cfg = PRESETS[preset_or_cfg] if isinstance(preset_or_cfg, str) else preset_or_cfg
    if max_seq_len is not None:
        cfg = ModelConfig(**{**cfg.to_dict(), "max_seq_len": max_seq_len})
    return Llama(cfg)
