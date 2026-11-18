"""Self-play training loop (paper Sec. 2) plus the uniform-prior and PCFG baselines.

Single process:   python scripts/train.py ...
Multi-GPU:        torchrun --standalone --nproc_per_node 8 scripts/train.py ...

With several processes the round pool is sharded across GPUs for the two expensive
steps (the forward-mode reward and the learner step, which uses DistributedDataParallel);
rank 0 owns the generator, the replay bank, the archive, the lookback checkpoints,
evaluation, logging and checkpointing, and broadcasts each round's outputs and
reward tangent to the other ranks.
"""
from __future__ import annotations

import json
import re
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.backends.cuda.matmul.allow_tf32 = True   # reward JVP: 25% faster, corr 0.999999 vs fp32
torch.backends.cudnn.allow_tf32 = True

from .bank import EliteArchive, Program, ReplayBank
from .bf import run_programs
from .data import (arithmetic_records, evaluate_bpb, load_byte_file, random_bytes_records,
                   records_from_bytes, sample_pcfg_rows, sample_uniform_programs)
from .fastattn import JVP_FLASH_AVAILABLE
from .generator import ProgramGenerator, generator_losses, log_prior
from .icl import evaluate_icl
from .model import build_model, per_sequence_cross_entropy
from .reward import adamw_preconditioner, directional_losses, make_tangent
from .structure import classify_output
from .tokens import START_OUTPUT


@dataclass
class TrainConfig:
    out_dir: str = "runs/debug"
    preset: str = "1m"
    data: str = "selfplay"          # selfplay | uniform | pcfg
    ctx: int = 4096                 # learner context; outputs have T = ctx - 1 bytes
    rounds: int = 1000
    seed: int = 0
    device: str = "cuda"
    # program space (App. E)
    max_program_len: int = 64
    tape_size: int = 1024
    step_budget: int = 65536
    # pool (App. G)
    n_programs: int = 1024          # N: fresh samples, a fraction of which are replaced by mutations
    mut_frac: float = 0.125
    n_replay: int = 256
    bank_capacity: int = 1_000_000
    archive_per_niche: int = 8
    archive_decay: float = 0.97
    # learner optimisation
    learner_lr: float = 3e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_rounds: int = 100
    grad_clip: float = 1.0
    micro_batch: int = 64
    # generator optimisation (Eqs. 3-5)
    gen_lr_ratio: float = 1.0
    beta_kl: float = 0.05
    lambda_ei: float = 1.0
    rho_clip: float = 20.0
    reward_variant: str = "abs"     # abs | signed | negate | shuffle | loss_delta
    lookback: str = "half"          # half: floor(e/2) | last_step: e-1
    reward_chunk: int = 16
    jvp_kernel: str = "auto"        # auto | flash (jvp_flash_attention package) | math
    lookback_storage: str = "disk"  # disk | memory
    # logging / evaluation
    log_every: int = 1
    eval_every: int = 100
    eval_ctx: int = 0               # 0 -> same as ctx
    eval_records: int = 256
    eval_ctx_extra: int = 256       # also evaluate at this record length with 4x the records (0 = off)
    eval_files: dict = field(default_factory=dict)   # name -> path of a raw byte file
    icl_every: int = 0
    icl_instances: int = 64
    ckpt_every: int = 500
    sample_every: int = 256         # retain the pool (programs + rewards) every k rounds
    wandb_project: str = ""         # non-empty -> also log every round to Weights & Biases
    wandb_name: str = ""            # run name (default: basename of out_dir)
    wandb_entity: str = ""
    gcs_path: str = ""              # e.g. gs://bucket/prefix -> checkpoints (learner + generator) are uploaded there

    @property
    def T(self) -> int:
        return self.ctx - 1


class LookbackStore:
    """Learner checkpoints theta_k, kept until no future round needs them."""

    def __init__(self, directory: Path | None):
        self.dir = directory
        self.mem: dict[int, dict[str, torch.Tensor]] = {}
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def save(self, k: int, params: dict[str, torch.Tensor]) -> None:
        cpu = {n: p.detach().to("cpu", copy=True) for n, p in params.items()}
        if self.dir is None:
            self.mem[k] = cpu
        else:
            torch.save(cpu, self.dir / f"theta_{k}.pt")

    def load(self, k: int) -> dict[str, torch.Tensor]:
        if self._has_exact(k):
            return self.mem[k] if self.dir is None else torch.load(self.dir / f"theta_{k}.pt")
        r = self.fallback_round(k)
        if r is None:
            raise KeyError(k)
        return self._load_fallback(r)

    def has(self, k: int) -> bool:
        return self._has_exact(k) or self.fallback_round(k) is not None

    def source_round(self, k: int) -> int:
        """The round whose weights load(k) returns (k itself, or the snapshot substituted for it)."""
        return k if self._has_exact(k) else self.fallback_round(k)

    def drop(self, k: int) -> None:
        if self.dir is None:
            self.mem.pop(k, None)
        else:
            p = self.dir / f"theta_{k}.pt"
            if p.exists():
                p.unlink()

    def prune(self, finished_round: int, keep_last: bool) -> None:
        """theta_k is needed by rounds 2k and 2k+1 (and by e-1 for last_step)."""
        e = finished_round
        for k in list(self.keys()):
            needed = (2 * k + 1 > e) or (keep_last and k >= e)
            if not needed:
                self.drop(k)

    # -- fallback after a resume: the store lives on the local SSD, which a preempted spot VM loses.
    # theta_k is then approximated by the learner weights of the newest mirrored weights-only snapshot
    # (models_round_XXXXXX.pt) with round <= k.
    def set_fallback(self, snapshot_dir: Path, param_names) -> None:
        import re
        self.fb_dir = Path(snapshot_dir)
        self.fb_names = set(param_names)
        self.fb_rounds = sorted(int(m.group(1)) for p in self.fb_dir.glob("models_round_*.pt")
                                for m in [re.search(r"models_round_(\d+)\.pt$", p.name)] if m)
        self.fb_cache: tuple[int, dict] | None = None

    def fallback_round(self, k: int):
        rs = [r for r in getattr(self, "fb_rounds", []) if r <= k]
        return rs[-1] if rs else None

    def _has_exact(self, k: int) -> bool:
        return k in self.mem if self.dir is None else (self.dir / f"theta_{k}.pt").exists()

    def _load_fallback(self, r: int) -> dict[str, torch.Tensor]:
        if self.fb_cache is not None and self.fb_cache[0] == r:
            return self.fb_cache[1]
        sd = torch.load(self.fb_dir / f"models_round_{r:06d}.pt", map_location="cpu", weights_only=False)["learner"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        params = {k: v for k, v in sd.items() if k in self.fb_names}
        self.fb_cache = (r, params)
        return params

    def keys(self):
        if self.dir is None:
            return list(self.mem.keys())
        return [int(p.stem.split("_")[1]) for p in self.dir.glob("theta_*.pt")]


def init_distributed() -> tuple[int, int, int]:
    """(rank, world_size, local_rank); initialises NCCL when launched by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return dist.get_rank(), dist.get_world_size(), local
    return 0, 1, 0


class SelfPlayTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rank, self.world, self.local_rank = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        self.out = Path(cfg.out_dir)
        if self.is_main:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

        self.learner = build_model(cfg.preset, cfg.ctx).to(self.device)
        self.ddp = DDP(self.learner, device_ids=[self.local_rank]) if self.world > 1 else None
        self.learner_opt = torch.optim.AdamW(self.learner.parameters(), lr=cfg.learner_lr,
                                             betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps,
                                             weight_decay=cfg.weight_decay)
        self.selfplay = cfg.data == "selfplay"
        if self.selfplay and self.is_main:
            self.gen_model = build_model(cfg.preset, cfg.max_program_len + 2).to(self.device)
            self.gen = ProgramGenerator(self.gen_model, cfg.max_program_len, self.device)
            self.gen_opt = torch.optim.AdamW(self.gen_model.parameters(), lr=cfg.learner_lr * cfg.gen_lr_ratio,
                                             betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps,
                                             weight_decay=cfg.weight_decay)
            self.bank = ReplayBank(cfg.bank_capacity, seed=cfg.seed)
            self.archive = EliteArchive(cfg.archive_per_niche, cfg.archive_decay, seed=cfg.seed)
            self.lookback = LookbackStore(None if cfg.lookback_storage == "memory" else self.out / "lookback")
            self.lookback.save(0, self.params())
        self.round = 0
        self.tokens_seen = 0
        self.log_path = self.out / "log.jsonl"
        self.eval_sets = self._build_eval_sets((cfg.eval_ctx or cfg.ctx) - 1, cfg.eval_records) if self.is_main else {}
        self.eval_sets_short = (self._build_eval_sets(cfg.eval_ctx_extra - 1, 4 * cfg.eval_records)
                                if (self.is_main and 0 < cfg.eval_ctx_extra < cfg.ctx) else {})
        self._last_grad_norm = 0.0
        self._t0 = time.time()
        self._upload: subprocess.Popen | None = None
        if cfg.gcs_path and shutil.which("gcloud") is None:
            raise RuntimeError("gcs_path set but `gcloud` is not on PATH")
        self.wandb = None
        if cfg.wandb_project and self.is_main:
            import wandb
            run_name = cfg.wandb_name or self.out.name
            self.wandb = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity or None,
                                    name=run_name, id=re.sub(r"[^A-Za-z0-9_-]", "-", run_name),
                                    config=asdict(cfg), dir=str(self.out), resume="allow")
            self.wandb.summary["learner_params"] = self.learner.num_params()
            self.wandb.summary["world_size"] = self.world

    # ------------------------------------------------------------------ helpers
    def params(self) -> dict[str, torch.Tensor]:
        return dict(self.learner.named_parameters())

    def buffers(self) -> dict[str, torch.Tensor]:
        return dict(self.learner.named_buffers())

    def lr_at(self, e: int) -> float:
        warm = max(self.cfg.warmup_rounds, 1)
        return self.cfg.learner_lr * min(1.0, (e + 1) / warm)

    def _build_eval_sets(self, L: int, n: int) -> dict[str, np.ndarray]:
        sets = {"random_bytes": random_bytes_records(n, L, seed=1),
                "arithmetic": arithmetic_records(n, L, seed=1)}
        for name, path in self.cfg.eval_files.items():
            sets[name] = records_from_bytes(load_byte_file(path), L, max_records=n)
        return sets

    def log(self, rec: dict, wandb_extra: dict | None = None) -> None:
        """rec goes to log.jsonl and W&B; wandb_extra (histograms, tables) only to W&B."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wandb is not None:
            self.wandb.log({**rec, **(wandb_extra or {})}, step=rec["round"])

    # ------------------------------------------------------------------ distributed plumbing
    def broadcast_outputs(self, outputs: np.ndarray | None) -> torch.Tensor:
        """Rank 0's [M, T] uint8 outputs -> learner sequences [M, ctx] (O prefix) on every rank."""
        if self.world == 1:
            out = torch.from_numpy(np.ascontiguousarray(outputs)).to(self.device)
        else:
            shape = [list(outputs.shape)] if self.is_main else [None]
            dist.broadcast_object_list(shape, src=0)
            if self.is_main:
                out = torch.from_numpy(np.ascontiguousarray(outputs)).to(self.device)
            else:
                out = torch.empty(shape[0], dtype=torch.uint8, device=self.device)
            dist.broadcast(out, src=0)
        prefix = torch.full((out.shape[0], 1), START_OUTPUT, dtype=torch.long, device=self.device)
        return torch.cat([prefix, out.long()], 1)

    def shard_bounds(self, M: int) -> tuple[int, int]:
        sizes = [len(t) for t in torch.arange(M).tensor_split(self.world)]
        start = sum(sizes[: self.rank])
        return start, start + sizes[self.rank]

    def gather(self, local: torch.Tensor, M: int) -> torch.Tensor:
        """Concatenate per-rank shards (in rank order) into a full [M] tensor on every rank."""
        if self.world == 1:
            return local
        sizes = [len(t) for t in torch.arange(M).tensor_split(self.world)]
        mx = max(sizes)
        buf = torch.zeros(mx, dtype=local.dtype, device=self.device)
        buf[: len(local)] = local
        parts = [torch.zeros_like(buf) for _ in range(self.world)]
        dist.all_gather(parts, buf)
        return torch.cat([p[:n] for p, n in zip(parts, sizes)])

    def broadcast_tangent(self, tangent: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
        names = list(self.params().keys())
        if self.world == 1:
            return tangent
        shapes = [self.params()[n].shape for n in names]
        numels = [s.numel() for s in shapes]
        if self.is_main:
            flat = torch.cat([tangent[n].reshape(-1).float() for n in names])
        else:
            flat = torch.zeros(sum(numels), device=self.device)
        dist.broadcast(flat, src=0)
        out, i = {}, 0
        for n, s, k in zip(names, shapes, numels):
            out[n] = flat[i : i + k].view(s)
            i += k
        return out

    # ------------------------------------------------------------------ pool (rank 0)
    def build_pool(self, e: int) -> tuple[list[Program], list[bool]]:
        cfg = self.cfg
        n_mut = int(cfg.n_programs * cfg.mut_frac) if e > 0 else 0
        mut_bodies = self.archive.propose(n_mut, cfg.max_program_len)
        n_fresh = cfg.n_programs - len(mut_bodies)
        fresh_bodies, fresh_logp = self.gen.sample(n_fresh)
        fresh = [Program(b, float(lp), e, "fresh") for b, lp in zip(fresh_bodies, fresh_logp.tolist())]
        mut = [Program(b, 0.0, e, "mut") for b in mut_bodies]
        replay = self.bank.sample(cfg.n_replay) if e > 0 else []
        pool = fresh + mut + replay
        return pool, [p.source == "mut" for p in pool]

    def execute(self, bodies: list[bytes], e: int):
        cfg = self.cfg
        return run_programs(bodies, tape_size=cfg.tape_size, step_budget=cfg.step_budget, T=cfg.T,
                            seed=cfg.seed * 1_000_003 + e)

    # ------------------------------------------------------------------ steps
    def make_round_tangent(self, e: int) -> tuple[dict[str, torch.Tensor], dict]:
        """Rank 0: v = P_e * (theta_past - theta_e), zero when no lookback checkpoint exists yet."""
        cfg = self.cfg
        now = self.params()
        past_round = (e - 1) if cfg.lookback == "last_step" else e // 2
        if e == 0 or not self.lookback.has(past_round):
            return {k: torch.zeros_like(v) for k, v in now.items()}, {"reward/delta_theta_norm": 0.0, "reward/past_round": past_round}
        past = self.lookback.load(past_round)
        P = adamw_preconditioner(self.learner_opt, now)
        tangent = make_tangent(P, past, now)
        delta_norm = math.sqrt(sum(float((past[k].to(v.device) - v.detach()).pow(2).sum()) for k, v in now.items()))
        return tangent, {"reward/delta_theta_norm": delta_norm, "reward/past_round": past_round,
                         "reward/lookback_from": self.lookback.source_round(past_round)}

    def rewards_from_directional(self, d: torch.Tensor) -> torch.Tensor:
        v = self.cfg.reward_variant
        if v in ("abs", "shuffle"):
            r = d.abs()
        elif v == "signed":
            r = d
        elif v == "negate":
            r = -d.abs()
        else:
            raise ValueError(v)
        if v == "shuffle":
            r = r[torch.randperm(len(r), device=r.device)]
        return r

    def shard_losses(self, shard: torch.Tensor, tangent: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return directional_losses(self.learner, self.params(), self.buffers(), tangent, shard,
                                  self.cfg.reward_chunk, self.cfg.jvp_kernel)

    def learner_step(self, shard: torch.Tensor, M: int, e: int) -> float:
        """One AdamW step on the full pool; this rank processes `shard` (M = pool size)."""
        cfg = self.cfg
        lr = self.lr_at(e)
        for g in self.learner_opt.param_groups:
            g["lr"] = lr
        model = self.ddp if self.ddp is not None else self.learner
        model.train()
        self.learner_opt.zero_grad(set_to_none=True)
        scale = self.world / M          # DDP averages over ranks -> global mean over the pool
        total = torch.zeros((), device=self.device)
        starts = list(range(0, shard.shape[0], cfg.micro_batch))
        for j, s in enumerate(starts):
            x = shard[s:s + cfg.micro_batch]
            ctx = model.no_sync() if (self.ddp is not None and j < len(starts) - 1) else _nullctx()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits = model(x[:, :-1])
                loss = per_sequence_cross_entropy(logits, x[:, 1:]).sum() * scale
                loss.backward()
            total += loss.detach()
        if self.ddp is not None and not starts:   # empty shard: still take part in the all-reduce
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = model(shard.new_full((1, 2), START_OUTPUT))
            (logits.sum() * 0.0).backward()
        gn = torch.nn.utils.clip_grad_norm_(self.learner.parameters(), cfg.grad_clip)
        self._last_grad_norm = float(gn)
        self.learner_opt.step()
        self.tokens_seen += M * cfg.T
        if self.world > 1:
            dist.all_reduce(total)
        return float(total) / self.world

    def generator_step(self, pool: list[Program], rewards: torch.Tensor, is_mut: torch.Tensor, e: int
                       ) -> tuple[dict, torch.Tensor]:
        cfg = self.cfg
        for g in self.gen_opt.param_groups:
            g["lr"] = self.lr_at(e) * cfg.gen_lr_ratio
        bodies = [p.body for p in pool]
        self.gen_model.train()
        self.gen_opt.zero_grad(set_to_none=True)
        logp = self.gen.log_probs(bodies)
        logp_old = torch.tensor([p.logp_old for p in pool], device=self.device)
        logp_old = torch.where(is_mut, logp.detach(), logp_old)     # mut rows: no sampling prob
        lprior = log_prior(bodies, self.device)
        loss, stats = generator_losses(logp, logp_old, lprior, rewards, is_mut, beta=cfg.beta_kl,
                                       lambda_ei=cfg.lambda_ei, rho_clip=cfg.rho_clip)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), cfg.grad_clip)
        self.gen_opt.step()
        stats["gen/grad_norm"] = float(gn)
        stats["gen/loss"] = float(loss)
        lens = torch.tensor([len(b) + 1 for b in bodies], device=self.device, dtype=torch.float32)
        stats["gen/nats_per_token"] = float(-(logp.detach() / lens).mean())
        stats["gen/frac_max_len"] = float(np.mean([len(b) >= cfg.max_program_len for b in bodies]))
        return stats, logp.detach()

    # ------------------------------------------------------------------ one round
    def selfplay_round(self, e: int) -> dict | None:
        cfg = self.cfg
        t0 = time.time()
        pool = ex = is_mut = None
        if self.is_main:
            pool, is_mut_list = self.build_pool(e)
            ex = self.execute([p.body for p in pool], e)
            is_mut = torch.tensor(is_mut_list, device=self.device)
        seqs = self.broadcast_outputs(ex.outputs if self.is_main else None)
        M = seqs.shape[0]
        s0, s1 = self.shard_bounds(M)
        shard = seqs[s0:s1]
        t_exec = time.time()

        zero = None
        if cfg.reward_variant == "loss_delta":
            zero = {k: torch.zeros_like(v) for k, v in self.params().items()}
            pre, _ = self.shard_losses(shard, zero)
            learner_loss = self.learner_step(shard, M, e)
            post, _ = self.shard_losses(shard, zero)
            rewards, losses, rstats = self.gather(pre - post, M), self.gather(pre, M), {}
            t_reward = t_exec
        else:
            tangent, rstats = self.make_round_tangent(e) if self.is_main else (None, {})
            tangent = self.broadcast_tangent(tangent)
            losses_l, d_l = self.shard_losses(shard, tangent)
            losses, d = self.gather(losses_l, M), self.gather(d_l, M)
            t_reward = time.time()
            rewards = self.rewards_from_directional(d)
            learner_loss = self.learner_step(shard, M, e)
        t_learner = time.time()
        if not self.is_main:
            return None

        gstats, logp_now = self.generator_step(pool, rewards, is_mut, e)
        t_gen = time.time()

        # bookkeeping: archive, replay bank, lookback checkpoints
        self.archive.step_decay()
        r_cpu = rewards.tolist()
        for p, r, d_ in zip(pool, r_cpu, ex.max_depth.tolist()):
            self.archive.add(p.body, r, d_, e)
        new_programs = []
        for p, lp in zip(pool, logp_now.tolist()):
            if p.source == "fresh":
                new_programs.append(p)
            elif p.source == "mut":
                new_programs.append(Program(p.body, lp, e, "mut"))
        self.bank.add(new_programs)
        self.lookback.save(e + 1, self.params())
        self.lookback.prune(e, keep_last=cfg.lookback == "last_step")

        wandb_extra: dict = {}
        if cfg.sample_every and e % cfg.sample_every == 0:
            fam_counts, table = self.dump_pool(e, pool, ex, r_cpu)
            self._pending_struct = {f"structure/{k}": v for k, v in fam_counts.items()}
            if self.wandb is not None:
                import wandb
                wandb_extra["programs/top_by_reward"] = wandb.Table(
                    columns=["round", "body", "source", "reward", "n_emitted", "max_depth", "family", "output_head"],
                    data=table)

        lens = np.array([len(p.body) for p in pool])
        src = np.array([p.source for p in pool])
        r_np = rewards.float().cpu().numpy()
        rec = {
            "round": e, "tokens": self.tokens_seen, "lr": self.lr_at(e), "pool": len(pool),
            "n_mut": int(is_mut.sum()), "n_replay": sum(p.source == "replay" for p in pool),
            "learner/loss": learner_loss, "learner/pool_loss_nats": float(losses.mean()),
            "learner/grad_norm": self._last_grad_norm,
            "learner/param_norm": math.sqrt(sum(float(p.detach().pow(2).sum()) for p in self.learner.parameters())),
            "reward/mean": float(rewards.mean()), "reward/std": float(rewards.std()),
            "reward/max": float(rewards.max()), "reward/frac_pos": float((rewards > 0).float().mean()),
            "reward/fresh_mean": float(r_np[src == "fresh"].mean()) if (src == "fresh").any() else 0.0,
            "reward/mut_mean": float(r_np[src == "mut"].mean()) if (src == "mut").any() else 0.0,
            "reward/replay_mean": float(r_np[src == "replay"].mean()) if (src == "replay").any() else 0.0,
            "loss/fresh_mean": float(losses[torch.from_numpy(src == "fresh").to(losses.device)].mean()) if (src == "fresh").any() else 0.0,
            "loss/replay_mean": float(losses[torch.from_numpy(src == "replay").to(losses.device)].mean()) if (src == "replay").any() else 0.0,
            "prog/len_mean": float(lens.mean()), "prog/emitted_mean": float(ex.n_emitted.mean()),
            "prog/frac_full_output": float((ex.n_emitted >= cfg.T).mean()),
            "prog/frac_empty_output": float((ex.n_emitted == 0).mean()),
            "prog/depth_mean": float(ex.max_depth.mean()), "prog/steps_mean": float(ex.steps.mean()),
            "archive/niches": self.archive.n_occupied, "archive/programs": self.archive.n_programs,
            "archive/mean_reward": float(np.mean([el.reward for v in self.archive.niches.values() for el in v])) if self.archive.n_programs else 0.0,
            "bank/size": len(self.bank),
            "time/exec": t_exec - t0, "time/reward": t_reward - t_exec, "time/learner": t_learner - t_reward,
            "time/gen": t_gen - t_learner, "time/round": time.time() - t0, "time/elapsed": time.time() - self._t0,
            "time/tokens_per_s": M * cfg.T / max(time.time() - t0, 1e-6),
        }
        rec.update(rstats)
        rec.update(gstats)
        if getattr(self, "_pending_struct", None):
            rec.update(self._pending_struct)
            self._pending_struct = None
        if self.wandb is not None:
            import wandb
            wandb_extra["hist/reward"] = wandb.Histogram(r_np)
            wandb_extra["hist/program_len"] = wandb.Histogram(lens)
            wandb_extra["hist/bytes_emitted"] = wandb.Histogram(ex.n_emitted)
            wandb_extra["hist/loop_depth"] = wandb.Histogram(ex.max_depth)
            wandb_extra["hist/pool_loss_nats"] = wandb.Histogram(losses.float().cpu().numpy())
        self._wandb_extra = wandb_extra
        return rec

    def baseline_round(self, e: int) -> dict | None:
        cfg = self.cfg
        t0 = time.time()
        outputs, extra = None, {}
        if self.is_main:
            if cfg.data == "uniform":
                bodies, _ = sample_uniform_programs(cfg.n_programs + cfg.n_replay, cfg.max_program_len, self.np_rng)
                ex = self.execute(bodies, e)
                outputs = ex.outputs
                extra = {"prog/len_mean": float(np.mean([len(b) for b in bodies])),
                         "prog/emitted_mean": float(ex.n_emitted.mean())}
            elif cfg.data == "pcfg":
                outputs = sample_pcfg_rows(cfg.n_programs + cfg.n_replay, cfg.T, self.np_rng)
            else:
                raise ValueError(cfg.data)
        seqs = self.broadcast_outputs(outputs)
        M = seqs.shape[0]
        s0, s1 = self.shard_bounds(M)
        learner_loss = self.learner_step(seqs[s0:s1], M, e)
        if not self.is_main:
            return None
        rec = {"round": e, "tokens": self.tokens_seen, "lr": self.lr_at(e), "pool": M,
               "learner/loss": learner_loss, "learner/grad_norm": self._last_grad_norm,
               "time/round": time.time() - t0, "time/elapsed": time.time() - self._t0,
               "time/tokens_per_s": M * cfg.T / max(time.time() - t0, 1e-6)}
        rec.update(extra)
        self._wandb_extra = {}
        return rec

    def dump_pool(self, e: int, pool, ex, rewards) -> tuple[dict[str, int], list[list]]:
        """Write the pool with rewards and detected math families; return family counts and a top-32 table."""
        from .structure import FAMILIES
        d = self.out / "programs"
        d.mkdir(exist_ok=True)
        counts = {f: 0 for f in FAMILIES}
        rows = []
        with open(d / f"round_{e:06d}.jsonl", "w") as f:
            for i, p in enumerate(pool):
                fam = classify_output(ex.outputs[i], int(ex.n_emitted[i]))
                if fam:
                    counts[fam] += 1
                row = {"body": p.body.decode("latin-1"), "source": p.source, "reward": rewards[i],
                       "n_emitted": int(ex.n_emitted[i]), "max_depth": int(ex.max_depth[i]),
                       "family": fam, "output_head": ex.outputs[i, :16].tolist()}
                f.write(json.dumps(row) + "\n")
                rows.append(row)
        rows.sort(key=lambda r: -r["reward"])
        table = [[e, "S" + r["body"] + "F", r["source"], r["reward"], r["n_emitted"], r["max_depth"],
                  r["family"] or "", str(r["output_head"])] for r in rows[:32]]
        return counts, table

    # ------------------------------------------------------------------ evaluation / checkpoints (rank 0)
    def evaluate(self) -> dict:
        res = {}
        for name, recs in self.eval_sets.items():
            res[f"eval/{name}_bpb"] = evaluate_bpb(self.learner, recs, device=self.device)
        for name, recs in self.eval_sets_short.items():
            res[f"eval{self.cfg.eval_ctx_extra}/{name}_bpb"] = evaluate_bpb(self.learner, recs, device=self.device)
        return res

    def evaluate_icl(self) -> dict:
        r = evaluate_icl(self.learner, n_instances=self.cfg.icl_instances, ctx=self.cfg.ctx, device=self.device)
        out = {f"icl/{task}/m{m}": v["success"] for task, d in r.items() for m, v in d.items()}
        out.update({f"icl_p/{task}/m{m}": v["p_correct"] for task, d in r.items() for m, v in d.items()})
        return out

    def save_checkpoint(self, name: str = "last") -> Path:
        """Full resumable checkpoint (ckpt_<name>.pt) plus a weights-only snapshot of the
        learner and the generator (models_round_XXXXXX.pt); both are mirrored to GCS if configured."""
        state = {"round": self.round, "tokens_seen": self.tokens_seen, "config": asdict(self.cfg),
                 "learner": self.learner.state_dict(), "learner_opt": self.learner_opt.state_dict(),
                 "model_cfg": self.learner.cfg.to_dict()}
        models = {"round": self.round, "tokens_seen": self.tokens_seen, "config": asdict(self.cfg),
                  "learner": self.learner.state_dict(), "model_cfg": self.learner.cfg.to_dict()}
        if self.selfplay:
            state.update({"generator": self.gen_model.state_dict(), "gen_opt": self.gen_opt.state_dict(),
                          "bank": self.bank.state_dict(), "archive": self.archive.state_dict()})
            models.update({"generator": self.gen_model.state_dict(), "generator_cfg": self.gen_model.cfg.to_dict(),
                           "max_program_len": self.cfg.max_program_len})
        path = self.out / f"ckpt_{name}.pt"
        torch.save(state, path)
        mpath = self.out / f"models_round_{self.round:06d}.pt"
        torch.save(models, mpath)
        self.upload_to_gcs([path, mpath, self.out / "config.json", self.log_path])
        return path

    def upload_to_gcs(self, files: list[Path], wait: bool = False) -> None:
        """Mirror files to <gcs_path>/<run name>/ with gcloud, in the background (one upload at a time)."""
        if not self.cfg.gcs_path:
            return
        if self._upload is not None:
            self._upload.wait()
        dest = self.cfg.gcs_path.rstrip("/") + "/" + self.out.name + "/"
        cmd = ["gcloud", "storage", "cp", "-q", *[str(f) for f in files if f.exists()], dest]
        self._upload = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(self.out / "gcs_upload.err", "a"))
        if wait:
            self._upload.wait()

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.learner.load_state_dict(state["learner"])
        self.learner_opt.load_state_dict(state["learner_opt"])
        self.round, self.tokens_seen = state["round"], state["tokens_seen"]
        if self.selfplay and self.is_main and "generator" in state:
            self.gen_model.load_state_dict(state["generator"])
            self.gen_opt.load_state_dict(state["gen_opt"])
            self.bank.load_state_dict(state["bank"])
            self.archive.load_state_dict(state["archive"])
            # the lookback store is local-SSD state: keep what survived, add theta at the resumed round
            # (= the loaded learner) and fall back to mirrored snapshots for anything else.
            self.lookback.save(self.round, self.params())
            self.lookback.set_fallback(self.out, self.params().keys())
            missing = [k for k in range(self.round // 2, self.round) if not self.lookback._has_exact(k)]
            print(f"resumed {self.out.name} at round {self.round} ({self.tokens_seen:,} tokens); lookback: "
                  f"{len(missing)} of {max(self.round - self.round // 2, 0)} needed thetas missing, substituted from "
                  f"{len(self.lookback.fb_rounds)} snapshots (every {self.cfg.ckpt_every} rounds)", flush=True)
        if self.is_main and self.log_path.exists():
            # drop log rows written after this checkpoint by the interrupted segment
            rows = [l for l in self.log_path.read_text().splitlines() if l.strip()]
            keep = [l for l in rows if json.loads(l).get("round", -1) < self.round]
            if len(keep) != len(rows):
                self.log_path.write_text("\n".join(keep) + ("\n" if keep else ""))

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        cfg = self.cfg
        if self.is_main:
            kern = "flash (jvp_flash_attention)" if cfg.jvp_kernel == "flash" or (cfg.jvp_kernel == "auto" and JVP_FLASH_AVAILABLE) else "math SDPA"
            print(f"learner params: {self.learner.num_params():,}  device: {self.device}  world: {self.world}  "
                  f"data: {cfg.data}  reward kernel: {kern}", flush=True)
        for e in range(self.round, cfg.rounds):
            rec = self.selfplay_round(e) if self.selfplay else self.baseline_round(e)
            self.round = e + 1
            if not self.is_main:
                continue
            if cfg.eval_every and (e % cfg.eval_every == 0 or e == cfg.rounds - 1):
                rec.update(self.evaluate())
            if cfg.icl_every and (e % cfg.icl_every == 0 or e == cfg.rounds - 1):
                rec.update(self.evaluate_icl())
            if cfg.ckpt_every and (self.round % cfg.ckpt_every == 0 or self.round == cfg.rounds):
                self.save_checkpoint("last")
            self.log(rec, getattr(self, "_wandb_extra", None))
            if cfg.log_every and e % cfg.log_every == 0:
                evals = " ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in rec.items() if k.startswith("eval/"))
                msg = (f"round {e:5d} loss {rec['learner/loss']:.4f} "
                       + (f"r_mean {rec['reward/mean']:.3e} len {rec['prog/len_mean']:.1f} "
                          f"emit {rec['prog/emitted_mean']:.0f} niches {rec['archive/niches']} " if self.selfplay else "")
                       + f"t {rec['time/round']:.1f}s {evals}")
                print(msg, flush=True)
        if self.is_main:
            if self.round == cfg.rounds:
                print(f"done: {self.tokens_seen:,} tokens; checkpoints in {self.out}"
                      + (f" and {cfg.gcs_path}/{self.out.name}/" if cfg.gcs_path else ""), flush=True)
            if self._upload is not None:
                self._upload.wait()
            if self.wandb is not None:
                self.wandb.finish()
        if self.world > 1:
            dist.barrier()
            dist.destroy_process_group()


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def load_learner(ckpt_path: str | Path, device="cuda"):
    from .model import Llama, ModelConfig
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Llama(ModelConfig(**state["model_cfg"])).to(device)
    model.load_state_dict(state["learner"])
    model.eval()
    return model, state
