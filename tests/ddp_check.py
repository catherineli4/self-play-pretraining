"""Multi-GPU consistency check (run manually):
   torchrun --standalone --nproc_per_node 2 tests/ddp_check.py
Rank 0 computes the full-pool reward alone and compares it with the sharded+gathered reward."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spp.bf import run_programs
from spp.data import sample_uniform_programs
from spp.train import SelfPlayTrainer, TrainConfig

cfg = TrainConfig(out_dir="/mnt/localssd/spp-runs/ddpcheck", preset="tiny", ctx=256, n_programs=40, n_replay=0,
                  max_program_len=16, reward_chunk=8, lookback_storage="memory", eval_every=0, sample_every=0)
tr = SelfPlayTrainer(cfg)
rng = np.random.default_rng(0)
bodies, _ = sample_uniform_programs(cfg.n_programs, cfg.max_program_len, rng)
ex = run_programs(bodies, T=cfg.T, step_budget=2000, seed=1)
seqs = tr.broadcast_outputs(ex.outputs if tr.is_main else None)
M = seqs.shape[0]
torch.manual_seed(123)
tangent = {k: torch.randn_like(v) * 1e-2 for k, v in tr.params().items()} if tr.is_main else None
tangent = tr.broadcast_tangent(tangent)
s0, s1 = tr.shard_bounds(M)
l_local, d_local = tr.shard_losses(seqs[s0:s1], tangent)
losses, d = tr.gather(l_local, M), tr.gather(d_local, M)
if tr.is_main:
    l_full, d_full = tr.shard_losses(seqs, tangent)
    print(f"world={tr.world}  max|loss diff|={float((losses - l_full).abs().max()):.2e}  "
          f"max|dirderiv diff|={float((d - d_full).abs().max()):.2e}  (|d| mean {float(d_full.abs().mean()):.2e})")
    assert torch.allclose(losses, l_full, atol=1e-4) and torch.allclose(d, d_full, atol=1e-4 * float(d_full.abs().mean()) + 1e-6)
    print("sharded reward == single-process reward: OK")
# learner step equivalence: compare parameters after one sharded step vs. a fresh single-process step
before = {k: v.detach().clone() for k, v in tr.params().items()}
tr.learner_step(seqs[s0:s1], M, 0)
after = {k: v.detach().clone() for k, v in tr.params().items()}
if tr.is_main:
    tr.learner.load_state_dict(before, strict=False)
    tr.learner_opt.state.clear()
    tr.world_backup, tr.ddp_backup, tr.rank_backup = tr.world, tr.ddp, tr.rank
    tr.world, tr.ddp = 1, None
    tr.learner_step(seqs, M, 0)
    single = {k: v.detach().clone() for k, v in tr.params().items()}
    diff = max(float((after[k] - single[k]).abs().max()) for k in after)
    print(f"max|param diff| sharded vs single-process learner step = {diff:.2e}")
    assert diff < 1e-4, diff
    print("sharded learner step == single-process learner step: OK")
    tr.world, tr.ddp = tr.world_backup, tr.ddp_backup
if tr.world > 1:
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
