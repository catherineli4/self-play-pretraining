"""Generator: sampling programs with restricted logits, sequence log-probs, and the
policy-gradient + expert-iteration losses (paper Eqs. 3-5)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .tokens import ALPHABET_SIZE, END_ID, PROGRAM_TOKEN_IDS, START_PROGRAM

LOG_ALPHABET = math.log(ALPHABET_SIZE)


def log_prior(bodies: list[bytes], device=None) -> torch.Tensor:
    """log g_0(x) = -ell(x) log|A|, ell counts tokens up to and including F."""
    lens = torch.tensor([len(b) + 1 for b in bodies], dtype=torch.float32, device=device)
    return -lens * LOG_ALPHABET


class ProgramGenerator:
    def __init__(self, model: torch.nn.Module, max_len: int, device: torch.device | str = "cuda"):
        self.model = model
        self.max_len = max_len
        self.device = torch.device(device)
        self.alphabet = torch.tensor(PROGRAM_TOKEN_IDS, device=self.device)
        self.end_index = PROGRAM_TOKEN_IDS.index(END_ID)
        lut = torch.full((256,), -1, dtype=torch.long)
        lut[self.alphabet.cpu()] = torch.arange(ALPHABET_SIZE)
        self.lut = lut.to(self.device)   # byte id -> alphabet index

    def restricted_log_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """logits [..., 256] -> log-probs over the 19 program tokens [..., 19]."""
        return F.log_softmax(logits[..., self.alphabet].float(), dim=-1)

    @torch.no_grad()
    def sample(self, n: int, temperature: float = 1.0, batch_size: int = 1024,
               ) -> tuple[list[bytes], torch.Tensor]:
        """Sample n programs; returns bodies and their sequence log-probs log g_phi(x)."""
        bodies, logps = [], []
        self.model.eval()
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            toks = torch.full((b, self.max_len + 2), START_PROGRAM, dtype=torch.long, device=self.device)
            alive = torch.ones(b, dtype=torch.bool, device=self.device)
            lengths = torch.full((b,), self.max_len, dtype=torch.long, device=self.device)
            logp = torch.zeros(b, device=self.device)
            for t in range(self.max_len + 1):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = self.model(toks[:, : t + 1])[:, -1]
                lp = self.restricted_log_softmax(logits / temperature)
                if t == self.max_len:  # body is full: F is forced with probability 1
                    idx = torch.full((b,), self.end_index, device=self.device)
                    step_lp = torch.zeros(b, device=self.device)
                else:
                    idx = torch.multinomial(lp.exp(), 1).squeeze(-1)
                    step_lp = lp.gather(-1, idx[:, None]).squeeze(-1)
                tok = self.alphabet[idx]
                toks[:, t + 1] = torch.where(alive, tok, toks[:, t + 1])
                logp = logp + torch.where(alive, step_lp, torch.zeros_like(step_lp))
                ended = alive & (tok == END_ID)
                lengths = torch.where(ended, torch.full_like(lengths, t), lengths)
                alive = alive & ~ended
                if not alive.any():
                    break
            toks_cpu, lens_cpu = toks.cpu(), lengths.cpu()
            for i in range(b):
                bodies.append(bytes(toks_cpu[i, 1 : 1 + int(lens_cpu[i])].tolist()))
            logps.append(logp)
        self.model.train()
        return bodies, torch.cat(logps)

    def pack(self, bodies: list[bytes]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad S+body+F sequences. Returns tokens [n, L+2] and a target mask [n, L+1]."""
        n = len(bodies)
        toks = torch.full((n, self.max_len + 2), END_ID, dtype=torch.long)
        mask = torch.zeros((n, self.max_len + 1), dtype=torch.bool)
        toks[:, 0] = START_PROGRAM
        for i, b in enumerate(bodies):
            L = len(b)
            toks[i, 1 : 1 + L] = torch.frombuffer(bytearray(b), dtype=torch.uint8).long() if L else toks[i, 1:1]
            toks[i, 1 + L] = END_ID
            mask[i, :L] = True
            mask[i, L] = L < self.max_len   # forced F carries no probability mass
        return toks.to(self.device), mask.to(self.device)

    def log_probs(self, bodies: list[bytes], batch_size: int = 512) -> torch.Tensor:
        """Differentiable sequence log-probs log g_phi(x_i) for a list of bodies."""
        out = []
        for start in range(0, len(bodies), batch_size):
            toks, mask = self.pack(bodies[start:start + batch_size])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = self.model(toks[:, :-1])
            lp = self.restricted_log_softmax(logits)                    # [b, L+1, 19]
            tgt = self.lut[toks[:, 1:]]                                 # alphabet indices
            tok_lp = lp.gather(-1, tgt.clamp_min(0)[..., None]).squeeze(-1)
            out.append((tok_lp * mask).sum(-1))
        return torch.cat(out)


def generator_losses(logp: torch.Tensor, logp_old: torch.Tensor, logprior: torch.Tensor,
                     rewards: torch.Tensor, is_mut: torch.Tensor, *, beta: float,
                     lambda_ei: float = 1.0, rho_clip: float = 20.0, eps: float = 1e-8,
                     gen_mask: torch.Tensor | None = None,
                     ) -> tuple[torch.Tensor, dict[str, float]]:
    """L_generator = L_PG + lambda_EI * L_EI over the round pool (paper Eqs. 4-5).

    gen_mask (bool, one per program) restricts both losses to a subset of the pool: the advantage z-scoring and the
    EI weights are computed within that subset, as if the other programs were never shown to the generator.
    None = the whole pool (PG on fresh + replayed programs, EI on everything, the paper's setting).
    """
    kl_all = float((logp.detach() - logprior).mean())
    n_update = int(logp.numel())
    if gen_mask is not None:
        idx = gen_mask.nonzero(as_tuple=True)[0]
        logp, logp_old, logprior, rewards, is_mut = (t[idx] for t in (logp, logp_old, logprior, rewards, is_mut))
        n_update = int(idx.numel())
    r = rewards.float()
    adv = (r - r.mean()) / (r.std(unbiased=False) + eps) - beta * (logp.detach() - logprior)
    log_rho = (logp.detach() - logp_old).clamp(-rho_clip, rho_clip)
    rho = log_rho.exp()
    pg_mask = ~is_mut
    n_pg = pg_mask.sum().clamp_min(1)
    loss_pg = -(rho * logp * adv * pg_mask).sum() / n_pg
    rpos = r.clamp_min(0.0)
    w = rpos / rpos.sum().clamp_min(eps)
    loss_ei = -(w * logp).sum()
    loss = loss_pg + lambda_ei * loss_ei
    stats = {
        "gen/loss_pg": float(loss_pg), "gen/loss_ei": float(loss_ei),
        "gen/adv_mean": float(adv.mean()), "gen/kl_to_prior": kl_all, "gen/n_update": n_update,
        "gen/rho_mean": float(rho[pg_mask].mean()) if int(n_pg) else 1.0,
    }
    return loss, stats
