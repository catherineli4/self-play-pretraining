#!/usr/bin/env python
"""Train self-play pretraining (or a baseline). Every TrainConfig field is a CLI flag.

Examples:
  python scripts/train.py --out_dir runs/sp1m --preset 1m --rounds 2000
  python scripts/train.py --data uniform --out_dir runs/uniform1m
  python scripts/train.py --eval text=/path/to/file.txt --eval dna=/path/dna.bin
"""
import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spp.train import SelfPlayTrainer, TrainConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="JSON file with TrainConfig fields")
    p.add_argument("--eval", action="append", default=[], help="name=path of a raw byte file (repeatable)")
    p.add_argument("--resume", help="checkpoint to resume from, or 'auto' = <out_dir>/ckpt_last.pt if it exists")
    for f in fields(TrainConfig):
        if f.name == "eval_files":
            continue
        p.add_argument(f"--{f.name}", type=type(f.default) if f.default is not None else str, default=None)
    a = p.parse_args()
    kw = {}
    if a.config:
        kw.update(json.loads(Path(a.config).read_text()))
    for f in fields(TrainConfig):
        v = getattr(a, f.name, None)
        if v is not None:
            kw[f.name] = v
    kw.setdefault("eval_files", {})
    for item in a.eval:
        name, path = item.split("=", 1)
        kw["eval_files"][name] = path
    cfg = TrainConfig(**kw)
    tr = SelfPlayTrainer(cfg)
    if a.resume:
        path = tr.out / "ckpt_last.pt" if a.resume == "auto" else Path(a.resume)
        if path.exists():
            tr.load_checkpoint(path)
        elif a.resume != "auto":
            raise FileNotFoundError(path)
    tr.train()


if __name__ == "__main__":
    main()
