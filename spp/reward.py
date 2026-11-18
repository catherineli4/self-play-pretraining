"""Learning-progress reward (paper Eq. 2) via forward-mode automatic differentiation.

    r_i = | < grad_theta L(y_i; theta_e),  P_e * (theta_{p(e)} - theta_e) > |
    P_e = lr / (sqrt(v_hat_e) + eps)          (diagonal AdamW step operator)

The inner product of a per-sequence gradient with a fixed direction v is the
directional derivative of the per-sequence loss along v, so a single JVP of the
vector-valued per-sequence loss gives every r_i in one forward pass without
materializing any gradient.
"""
from __future__ import annotations

import torch
from torch.func import functional_call, jvp
from .fastattn import forward_mode_attention
from .model import per_sequence_cross_entropy


def adamw_preconditioner(optimizer: torch.optim.Optimizer, named_params: dict[str, torch.Tensor],
                         ) -> dict[str, torch.Tensor]:
    """P = lr / (sqrt(v_hat) + eps) per parameter, from the learner's AdamW state.

    Parameters with no optimizer state yet (before the first step) get v_hat = 0.
    """
    by_id = {id(p): name for name, p in named_params.items()}
    out: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        lr, eps = group["lr"], group["eps"]
        beta2 = group["betas"][1]
        for p in group["params"]:
            name = by_id.get(id(p))
            if name is None:
                continue
            state = optimizer.state.get(p, {})
            if "exp_avg_sq" in state and state.get("step", 0) is not None:
                step = state["step"]
                step = float(step.item()) if torch.is_tensor(step) else float(step)
                v_hat = state["exp_avg_sq"] / (1.0 - beta2 ** step) if step > 0 else state["exp_avg_sq"]
            else:
                v_hat = torch.zeros_like(p)
            out[name] = lr / (v_hat.sqrt() + eps)
    return out


def make_tangent(preconditioner: dict[str, torch.Tensor], theta_past: dict[str, torch.Tensor],
                 theta_now: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """v = P * (theta_past - theta_now)."""
    return {k: preconditioner[k] * (theta_past[k].to(theta_now[k].device) - theta_now[k].detach())
            for k in theta_now}


@torch.no_grad()
def directional_losses(model: torch.nn.Module, params: dict[str, torch.Tensor],
                       buffers: dict[str, torch.Tensor], tangent: dict[str, torch.Tensor],
                       seqs: torch.Tensor, chunk_size: int = 16, kernel: str = "auto",
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence loss and its directional derivative along `tangent`.

    seqs: [M, ctx] int64 (prefix byte + content). Returns (losses [M], dirderiv [M]).
    """
    losses, derivs = [], []
    params = {k: v.detach() for k, v in params.items()}

    for start in range(0, seqs.shape[0], chunk_size):
        x = seqs[start:start + chunk_size]

        def f(p):
            logits = functional_call(model, (p, buffers), (x,))
            return per_sequence_cross_entropy(logits[:, :-1], x[:, 1:])

        with forward_mode_attention(kernel):   # built-in flash/efficient kernels lack forward-mode AD
            loss, d = jvp(f, (params,), (tangent,))
        losses.append(loss)
        derivs.append(d)
    return torch.cat(losses), torch.cat(derivs)


def learning_progress_reward(model, params, buffers, tangent, seqs, *, chunk_size: int = 16,
                             variant: str = "abs") -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (rewards [M], losses [M]). variant: abs (canonical) | signed | negate."""
    losses, d = directional_losses(model, params, buffers, tangent, seqs, chunk_size)
    if variant == "abs":
        r = d.abs()
    elif variant == "signed":
        r = d
    elif variant == "negate":
        r = -d.abs()
    else:
        raise ValueError(f"unknown reward variant {variant}")
    return r, losses
