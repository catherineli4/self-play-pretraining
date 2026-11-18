#!/usr/bin/env python
"""Run the pre-pretraining sweep (one seed) across GPUs: from-scratch arms over initialization
scales x LR x WD, plus warm-start arms over LR x WD. Finished jobs (results.json) are skipped.

  python scripts/sweep_pretrain.py --corpus /mnt/localssd/spp-data/metamath.bin --out_root /mnt/localssd/spp-pp \
      --gpus 0,1,2,3,4,5,6 --sizes 1m --stds 0.005,0.01,0.02,0.03,0.05,0.1 \
      --warm_start 1m=/mnt/localssd/spp-runs/sp_1m_seed0/models_round_008192.pt
Each setting is a coordinate of the init-scale grid: --stds sweeps the global std with the other
scales at their defaults; --emb_stds, --resid_scales and --head_stds do the same for the other knobs
(at --base_std). Pass --wandb_project to log every run. `run_jobs` is reused by sweep_fresh.py.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_jobs(jobs, out_root: Path, gpus: str, per_gpu: int, common_args: list, wandb_project: str = "",
             dry_run: bool = False):
    """jobs: list of (name, args). Runs scripts/pretrain_natural.py per job, one per GPU slot, skipping
    jobs whose results.json exists. Prints start/done lines; returns the number of failures."""
    out_root.mkdir(parents=True, exist_ok=True)
    pending = [(n, a) for n, a in jobs if not (out_root / n / "results.json").exists()]
    print(f"{len(jobs)} jobs, {len(pending)} pending", flush=True)
    if dry_run:
        for n, a in pending:
            print(n, " ".join(a))
        return 0
    slots = [g for g in gpus.split(",") for _ in range(per_gpu)]
    running: dict = {}
    queue = list(pending)
    failures = 0
    t0 = time.time()
    while queue or running:
        for slot in list(slots):
            if slot in running or not queue:
                continue
            name, args = queue.pop(0)
            cmd = [PY, str(ROOT / "scripts/pretrain_natural.py"), "--out_dir", str(out_root / name), *common_args, *args]
            if wandb_project:
                cmd += ["--wandb_project", wandb_project, "--wandb_name", name, "--wandb_group", name.split("_lr")[0]]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": slot}
            (out_root / name).mkdir(parents=True, exist_ok=True)
            logf = open(out_root / name / "stdout.txt", "w")
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running[slot] = (proc, name, time.time())
            print(f"[{time.time()-t0:6.0f}s] start {name} on gpu {slot} ({len(queue)} queued)", flush=True)
        for slot, (proc, name, ts) in list(running.items()):
            if proc.poll() is not None:
                status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                failures += proc.returncode != 0
                res = out_root / name / "results.json"
                summary = ""
                if res.exists():
                    r = json.loads(res.read_text())
                    summary = (f" final {r['converged_bpb']:.4f} zero-shot {r['zero_shot_bpb']:.3f} "
                               f"tokens {r['tokens_total']/1e6:.0f}M ({r.get('stop_reason')})")
                print(f"[{time.time()-t0:6.0f}s] done  {name} {status} in {time.time()-ts:.0f}s{summary}", flush=True)
                del running[slot]
        time.sleep(2)
    print(f"sweep complete ({failures} failures)", flush=True)
    return failures


def jobs_from_args(a):
    jobs = []
    lrs = [float(x) for x in a.lrs.split(",")]
    wds = [float(x) for x in a.wds.split(",")]
    for size in a.sizes.split(","):
        settings = []
        for std in (a.stds.split(",") if a.stds else []):
            settings.append(("std", std, dict(init_std=std)))
        for e in (a.emb_stds.split(",") if a.emb_stds else []):
            settings.append(("emb", e, dict(init_std=a.base_std, init_emb_std=e)))
        for r in (a.resid_scales.split(",") if a.resid_scales else []):
            settings.append(("resid", r, dict(init_std=a.base_std, init_resid_scale=r)))
        for h in (a.head_stds.split(",") if a.head_stds else []):
            settings.append(("head", h, dict(init_std=a.base_std, init_head_std=h)))
        for (knob, val, init_kw), lr, wd in itertools.product(settings, lrs, wds):
            name = f"{size}_scratch_{knob}{val.replace('/', '_')}_lr{lr:g}_wd{wd:g}"
            args = ["--preset", size, "--lr", str(lr), "--weight_decay", str(wd)]
            for k, v in init_kw.items():
                args += [f"--{k}", str(v)]
            jobs.append((name, args))
        for ws in a.warm_start:
            ws_size, path = ws.split("=", 1)
            if ws_size != size:
                continue
            for lr, wd in itertools.product(lrs, wds):
                name = f"{size}_warm_lr{lr:g}_wd{wd:g}"
                jobs.append((name, ["--preset", size, "--lr", str(lr), "--weight_decay", str(wd), "--warm_start", path]))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--per_gpu", type=int, default=1)
    p.add_argument("--sizes", default="1m")
    p.add_argument("--stds", default="0.005,0.01,0.02,0.03,0.05,0.1")
    p.add_argument("--emb_stds", default="")
    p.add_argument("--resid_scales", default="")
    p.add_argument("--head_stds", default="")
    p.add_argument("--base_std", default="0.02")
    p.add_argument("--lrs", default="1e-3,3e-3,1e-2")
    p.add_argument("--wds", default="0,0.1,0.3,0.8")
    p.add_argument("--warm_start", action="append", default=[], help="size=path (repeatable)")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--wandb_project", default="")
    p.add_argument("--extra", default="", help="extra args passed to every run, e.g. '--eval_every 50'")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    common = ["--corpus", a.corpus, "--ctx", str(a.ctx), *a.extra.split()]
    run_jobs(jobs_from_args(a), Path(a.out_root), a.gpus, a.per_gpu, common, a.wandb_project, a.dry_run)


if __name__ == "__main__":
    main()
