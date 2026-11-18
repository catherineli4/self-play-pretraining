#!/usr/bin/env python
"""Pre-pretraining experiment (paper App. A): pretrain on a natural byte corpus from scratch
with a chosen initialization scale, or warm-started from a self-play learner checkpoint.

  python scripts/pretrain_natural.py --corpus data/metamath.bin --out_dir runs/pp/scratch_std0.02 \
      --preset 1m --init_std 0.02 --init_emb_std 0.02 --init_head_std 0.02 --init_resid_scale "1/sqrt(2L)"
  python scripts/pretrain_natural.py --corpus data/metamath.bin --out_dir runs/pp/warm \
      --preset 1m --warm_start /mnt/localssd/spp-runs/sp_1m_seed0/models_round_008192.pt
"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spp.init import InitScales  # noqa: E402
from spp.pretrain import PretrainConfig, run_pretraining  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    for f in fields(PretrainConfig):
        if f.name == "init":
            continue
        p.add_argument(f"--{f.name}", type=type(f.default) if f.default is not None else str, default=None)
    p.add_argument("--init_std", type=float, default=0.02)
    p.add_argument("--init_emb_std", type=float, default=0.02)
    p.add_argument("--init_head_std", type=float, default=0.02)
    p.add_argument("--init_resid_scale", type=str, default="1/sqrt(2L)", help='number or "1/sqrt(2L)"')
    p.add_argument("--init_seed", type=int, default=0)
    a = p.parse_args()
    kw = {f.name: getattr(a, f.name) for f in fields(PretrainConfig) if f.name != "init" and getattr(a, f.name) is not None}
    rs = a.init_resid_scale
    try:
        rs = float(rs)
    except ValueError:
        pass
    kw["init"] = InitScales(std=a.init_std, emb_std=a.init_emb_std, head_std=a.init_head_std,
                            resid_scale=rs, seed=a.init_seed)
    if not kw.get("corpus"):
        p.error("--corpus is required")
    run_pretraining(PretrainConfig(**kw))


if __name__ == "__main__":
    main()
