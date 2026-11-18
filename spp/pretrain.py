"""Natural-data pretraining following paper App. A ("pre-pretraining" experiment).

Two arms share this code: from scratch with a chosen initialization scale (InitScales),
or warm-started from a self-play learner checkpoint (models_round_XXXXXX.pt).

Protocols:
  * plateau (default): constant learning rate (short warmup) until validation bits-per-byte
    plateaus (improvement < plateau_tol for plateau_patience consecutive evaluations), then a
    cosine decay to zero over decay_steps; the reported number is the converged BPB after it.
  * fixed budget (max_epochs > 0): exactly max_epochs passes over the (optionally prefix-capped)
    corpus, the cosine decay occupying the last decay_steps optimizer steps; max_epochs = 1 with
    corpus_tokens = N is a fresh-token run on the first N bytes with no repetition.

Multi-GPU: launch with `torchrun --nproc_per_node K scripts/pretrain_natural.py ...`; the global batch
(batch_size records) is split evenly across ranks, gradients are averaged by DDP, every rank evaluates the
same validation set (so all ranks take the same decisions) and rank 0 logs, exports and reports.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from .data import evaluate_bpb, load_byte_file, records_from_bytes
from .init import InitScales, apply_init
from .model import build_model, per_sequence_cross_entropy
from .tokens import START_OUTPUT


@dataclass
class PretrainConfig:
    corpus: str = ""                # raw byte file (e.g. dclm.bin)
    out_dir: str = "runs/pretrain/debug"
    preset: str = "1m"
    ctx: int = 4096
    warm_start: str = ""            # self-play checkpoint; empty -> from scratch with `init`
    init: InitScales = field(default_factory=InitScales)
    lr: float = 3e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    batch_size: int = 64            # records per optimizer step (64 x 4095 = 262k tokens)
    micro_batch: int = 32
    warmup_steps: int = 100
    eval_every: int = 100           # optimizer steps between validation evaluations
    val_corpus: str = ""            # separate held-out byte file (its first val_max records); empty -> split `corpus`
    val_frac: float = 0.05
    val_min: int = 256
    val_max: int = 1024
    corpus_tokens: float = 0.0      # > 0: train only on the first N bytes of `corpus`
    max_epochs: float = 0.0         # > 0: fixed budget of this many passes over the training records (plateau rule off)
    plateau_tol: float = 0.005      # paper: improvement < 0.005 BPB ...
    plateau_patience: int = 5       # ... for 5 consecutive evaluations
    decay_steps: int = 200          # paper: 200-step cosine decay to zero
    max_tokens: float = 3e9         # plateau mode safety cap: forces the decay if no plateau by then
    seed: int = 0
    device: str = "cuda"
    log_every: int = 20
    wandb_project: str = ""
    wandb_name: str = ""
    wandb_group: str = ""
    save_at_tokens: str = ""        # comma-separated token counts; the learner is exported at each (warm-start format)
    save_final: int = 0             # 1: export the final learner as ckpt_final.pt (warm-start format)

    @property
    def T(self) -> int:
        return self.ctx - 1


def _records_to_batch(recs: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(recs)).long()
    prefix = torch.full((x.shape[0], 1), START_OUTPUT, dtype=torch.long)
    return torch.cat([prefix, x], 1).to(device)


def load_warm_start(model, path: str, device) -> dict:
    st = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(st["learner"])
    return {"warm_start_round": st.get("round"), "warm_start_tokens": st.get("tokens_seen")}


def run_pretraining(cfg: PretrainConfig) -> dict:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    is_main = rank == 0
    if cfg.batch_size % world:
        raise ValueError(f"batch_size {cfg.batch_size} must be divisible by the world size {world}")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    if is_main:
        (out / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    # data: fixed records; validation either from a separate file or a held-out split
    data = load_byte_file(cfg.corpus)
    if cfg.corpus_tokens > 0:
        data = data[: int(cfg.corpus_tokens)]
    recs = records_from_bytes(data, cfg.T)
    if cfg.val_corpus:
        val = records_from_bytes(load_byte_file(cfg.val_corpus), cfg.T)[: cfg.val_max]
        train = recs
    else:
        n_val = int(min(cfg.val_max, max(cfg.val_min, cfg.val_frac * len(recs))))
        perm = rng.permutation(len(recs))
        val, train = recs[perm[:n_val]], recs[perm[n_val:]]
    if len(train) == 0 or len(val) == 0:
        raise ValueError("corpus too small for the requested context")
    tokens_per_epoch = int(len(train) * cfg.T)
    total_steps = int(cfg.max_epochs * len(train) / cfg.batch_size) if cfg.max_epochs > 0 else None
    if total_steps is not None and total_steps <= cfg.warmup_steps + cfg.decay_steps:
        raise ValueError(f"budget of {total_steps} steps is too small for warmup + decay")

    # model
    model = build_model(cfg.preset, cfg.ctx).to(device)
    info = {"n_params": model.num_params(), "n_train_records": int(len(train)), "n_val_records": int(len(val)),
            "tokens_per_epoch": tokens_per_epoch, "total_steps": total_steps,
            "arm": "warm_start" if cfg.warm_start else "scratch"}
    if cfg.warm_start:
        info.update(load_warm_start(model, cfg.warm_start, device))
    else:
        info["init"] = apply_init(model, cfg.init)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps,
                            weight_decay=cfg.weight_decay)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    info["world_size"] = world

    wb = None
    if cfg.wandb_project and is_main:
        import wandb
        wb = wandb.init(project=cfg.wandb_project, name=cfg.wandb_name or out.name, group=cfg.wandb_group or None,
                        config={**cfg_dict, **info}, dir=str(out))

    log_f = open(out / "log.jsonl", "a") if is_main else None

    def log(rec):
        if not is_main:
            return
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if wb is not None:
            wb.log(rec, step=rec["step"])

    def evaluate():
        return evaluate_bpb(model, val, batch_size=cfg.micro_batch, device=device)

    step, tokens, epoch = 0, 0, 0
    phase, decay_start, best, stall = "constant", None, float("inf"), 0
    tokens_at_plateau, stop_reason, converged_bpb = None, None, None
    history = []
    t0 = time.time()
    bpb0 = evaluate()
    history.append({"step": 0, "tokens": 0, "val_bpb": bpb0, "phase": phase})
    log({"step": 0, "tokens": 0, "epoch_frac": 0.0, "val_bpb": bpb0, "lr": 0.0, "phase": phase, "epoch": 0})
    budget = f"{total_steps} steps ({cfg.max_epochs:g} epochs)" if total_steps else "until plateau"
    if is_main:
        print(f"[{out.name}] {info['arm']} {cfg.preset} params={info['n_params']:,} train_records={len(train)} "
              f"({tokens_per_epoch/1e6:.0f}M tokens/epoch) val_records={len(val)} budget={budget} "
              f"world={world} zero-shot val bpb {bpb0:.4f}", flush=True)

    def export(path: Path):
        if not is_main:
            return
        torch.save({"learner": model.state_dict(), "model_cfg": model.cfg.to_dict(), "round": None, "tokens_seen": tokens,
                    "source": {"corpus": cfg.corpus, "arm": info["arm"], "warm_start": cfg.warm_start, "lr": cfg.lr,
                               "step": step}}, path)
        print(f"[{out.name}] exported {path.name} at {tokens/1e6:.0f}M tokens", flush=True)

    milestones = sorted(float(x) for x in cfg.save_at_tokens.split(",") if x)
    order = rng.permutation(len(train))
    pos = 0
    while True:
        if total_steps is not None and phase == "constant" and step >= total_steps - cfg.decay_steps:
            phase, decay_start, tokens_at_plateau, stop_reason = "decay", step, tokens, "budget"
        # learning rate
        if phase == "constant":
            lr = cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
        else:
            frac = min(1.0, (step - decay_start) / max(cfg.decay_steps, 1))
            lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        # batch (epoch-wise shuffling)
        if pos + cfg.batch_size > len(order):
            order = rng.permutation(len(train))
            pos = 0
            epoch += 1
        idx = order[pos:pos + cfg.batch_size]
        pos += cfg.batch_size
        x = _records_to_batch(train[idx[rank::world]], device)      # this rank's share of the global batch
        net.train()
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for s in range(0, x.shape[0], cfg.micro_batch):
            xb = x[s:s + cfg.micro_batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = net(xb[:, :-1])
            loss = per_sequence_cross_entropy(logits, xb[:, 1:]).sum() / x.shape[0]
            loss.backward()
            total += loss.detach()
        if world > 1:
            dist.all_reduce(total, op=dist.ReduceOp.AVG)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        tokens += len(idx) * cfg.T
        while milestones and tokens >= milestones[0]:
            export(out / f"ckpt_tokens_{milestones.pop(0)/1e6:.0f}M.pt")

        rec = None
        if step % cfg.log_every == 0:
            rec = {"step": step, "tokens": tokens, "epoch_frac": tokens / tokens_per_epoch,
                   "train_loss_nats": float(total), "train_bpb": float(total) / math.log(2),
                   "lr": lr, "phase": phase, "epoch": epoch, "grad_norm": float(gn), "elapsed": time.time() - t0}
        decay_done = phase == "decay" and step - decay_start >= cfg.decay_steps
        if step % cfg.eval_every == 0 or decay_done:
            bpb = evaluate()
            rec = rec or {"step": step, "tokens": tokens, "epoch_frac": tokens / tokens_per_epoch, "lr": lr,
                          "phase": phase, "epoch": epoch, "elapsed": time.time() - t0}
            rec["val_bpb"] = bpb
            history.append({"step": step, "tokens": tokens, "val_bpb": bpb, "phase": phase})
            if phase == "constant":
                if best - bpb < cfg.plateau_tol:
                    stall += 1
                else:
                    stall = 0
                best = min(best, bpb)
                if total_steps is None and (stall >= cfg.plateau_patience or tokens >= cfg.max_tokens):
                    phase, decay_start, tokens_at_plateau = "decay", step, tokens
                    stop_reason = "plateau" if stall >= cfg.plateau_patience else "max_tokens"
                    rec["plateau"] = 1
                    if is_main:
                        print(f"[{out.name}] plateau at step {step} ({tokens/1e6:.0f}M tokens), val bpb {bpb:.4f}; decaying", flush=True)
            if decay_done:
                converged_bpb = bpb
        if rec is not None:
            log(rec)
        if converged_bpb is not None:
            break

    result = {**info, "zero_shot_bpb": bpb0, "best_constant_bpb": best, "converged_bpb": converged_bpb,
              "stop_reason": stop_reason, "tokens_at_plateau": tokens_at_plateau, "tokens_total": tokens,
              "steps": step, "epochs": epoch, "wall_s": time.time() - t0, "config": cfg_dict, "history": history}
    if is_main:
        (out / "results.json").write_text(json.dumps(result, indent=2))
    if cfg.save_final:
        export(out / "ckpt_final.pt")
    if wb is not None:
        wb.summary.update({k: v for k, v in result.items() if k not in ("config", "history")})
        wb.finish()
    if log_f is not None:
        log_f.close()
    if is_main:
        print(f"[{out.name}] final bpb {converged_bpb:.4f} after {tokens/1e6:.0f}M tokens "
              f"({tokens/tokens_per_epoch:.2f} epochs, stop: {stop_reason}), {time.time()-t0:.0f}s", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return result
