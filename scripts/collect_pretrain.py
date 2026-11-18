#!/usr/bin/env python
"""Summarise a pre-pretraining sweep: best (LR, WD) per arm/setting by converged BPB."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_root")
    p.add_argument("--csv", default="")
    a = p.parse_args()
    rows = []
    for res in sorted(Path(a.out_root).glob("*/results.json")):
        r = json.loads(res.read_text())
        name = res.parent.name
        m = re.match(r"(?P<size>[^_]+)_(?P<arm>scratch|warm)(?:_(?P<setting>[^_]+))?_lr(?P<lr>[^_]+)_wd(?P<wd>.+)$", name)
        if not m:
            continue
        d = m.groupdict()
        rows.append({**d, "setting": d["setting"] or "-", "name": name, "zero_shot": r["zero_shot_bpb"],
                     "converged": r["converged_bpb"], "best_const": r["best_constant_bpb"],
                     "tokens_plateau_M": (r["tokens_at_plateau"] or 0) / 1e6, "tokens_total_M": r["tokens_total"] / 1e6,
                     "wall_min": r["wall_s"] / 60})
    groups = defaultdict(list)
    for r in rows:
        groups[(r["size"], r["arm"], r["setting"])].append(r)
    print(f"| size | arm | setting | best lr | best wd | zero-shot bpb | converged bpb | tokens to converge (M) | runs |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    summary = []
    for key in sorted(groups):
        g = groups[key]
        b = min(g, key=lambda r: r["converged"])
        summary.append(b)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {b['lr']} | {b['wd']} | {b['zero_shot']:.3f} | {b['converged']:.4f} | "
              f"{b['tokens_total_M']:.0f} | {len(g)} |")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
