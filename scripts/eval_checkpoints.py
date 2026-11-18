#!/usr/bin/env python
"""Zero-shot bits-per-byte of every saved checkpoint of a run (models_round_*.pt) on byte files,
at the training context, written to <run>/eval_curve.jsonl (one row per checkpoint, skips rows already done).

  python scripts/eval_checkpoints.py /mnt/localssd/spp-runs/sp_1m_seed0 --eval dclm=/mnt/localssd/spp-data/dclm.bin
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spp.data import evaluate_bpb, load_byte_file, records_from_bytes  # noqa: E402
from spp.train import load_learner  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--eval", action="append", default=[], help="name=path (raw bytes), repeatable")
    p.add_argument("--records", type=int, default=256, help="records per eval set (from the END of the file)")
    p.add_argument("--ctx", type=int, default=0, help="0 = the checkpoint's training context")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run = Path(a.run_dir)
    out = run / "eval_curve.jsonl"
    done = {json.loads(l)["round"] for l in open(out)} if out.exists() else set()
    sets = {}
    for item in a.eval:
        name, path = item.split("=", 1)
        sets[name] = load_byte_file(path)
    for ck in sorted(glob.glob(str(run / "models_round_*.pt"))):
        model, st = load_learner(ck, a.device)
        if st["round"] in done:
            continue
        L = (a.ctx or model.cfg.max_seq_len) - 1
        rec = {"round": st["round"], "tokens": st["tokens_seen"]}
        for name, data in sets.items():
            recs = records_from_bytes(data, L)[-a.records:]
            rec[f"eval/{name}_bpb"] = evaluate_bpb(model, recs, device=a.device)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)


if __name__ == "__main__":
    main()
