#!/usr/bin/env python
"""Fresh-token, single-epoch pre-pretraining comparison (no data repetition): warm starts versus
from-scratch arms on the first N bytes of a DCLM stream, validated on a separate held-out DCLM file.
Each arm gets a small LR x WD grid per budget; the best final bpb per (budget, arm) is reported.

  python scripts/sweep_fresh.py --corpus /mnt/localssd/spp-data/dclm2b.bin --val_corpus /mnt/localssd/spp-data/dclm_val.bin \
      --out_root /mnt/localssd/spp-pp1m-fresh --gpus 0,1,2,3,4,5,6,7 \
      --warm_final /mnt/localssd/spp-runs/sp_1m_seed0/models_round_008192.pt \
      --warm_r2800 /mnt/localssd/spp-runs/sp_1m_seed0/models_round_002800.pt
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_pretrain import run_jobs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--val_corpus", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--per_gpu", type=int, default=1)
    p.add_argument("--preset", default="1m")
    p.add_argument("--budgets", default="1B=1e9,2B=2e9", help="name=bytes, comma-separated")
    p.add_argument("--scratch_stds", default="0.02,0.005", help="from-scratch init stds (0.02 = Llama default)")
    p.add_argument("--warm_final", default="", help="final self-play checkpoint (arm name warm_final)")
    p.add_argument("--warm_r2800", default="", help="round-2800 self-play checkpoint (arm name warm_r2800)")
    p.add_argument("--arm", action="append", default=[], help="NAME=checkpoint: an extra warm-start arm (repeatable)")
    p.add_argument("--lrs", default="3e-3,1e-2")
    p.add_argument("--wds", default="0,0.1")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--wandb_project", default="")
    p.add_argument("--extra", default="")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    arms = [(f"scratch_std{s}", ["--init_std", s]) for s in a.scratch_stds.split(",") if s]
    if a.warm_final:
        arms.append(("warm_final", ["--warm_start", a.warm_final]))
    if a.warm_r2800:
        arms.append(("warm_r2800", ["--warm_start", a.warm_r2800]))
    for item in a.arm:
        name, path = item.split("=", 1)
        arms.append((name, ["--warm_start", path]))
    budgets = [(b.split("=")[0], float(b.split("=")[1])) for b in a.budgets.split(",")]
    budgets.sort(key=lambda x: -x[1])          # longest jobs first for better packing
    lrs = [float(x) for x in a.lrs.split(",")]
    wds = [float(x) for x in a.wds.split(",")]
    jobs = []
    for (bname, nbytes), (arm, arm_args), lr, wd in itertools.product(budgets, arms, lrs, wds):
        name = f"{bname}_{arm}_lr{lr:g}_wd{wd:g}"
        jobs.append((name, ["--preset", a.preset, "--corpus_tokens", f"{nbytes:g}", "--max_epochs", "1",
                            "--lr", str(lr), "--weight_decay", str(wd), *arm_args]))
    common = ["--corpus", a.corpus, "--val_corpus", a.val_corpus, "--ctx", str(a.ctx), *a.extra.split()]
    failures = run_jobs(jobs, Path(a.out_root), a.gpus, a.per_gpu, common, a.wandb_project, a.dry_run)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
