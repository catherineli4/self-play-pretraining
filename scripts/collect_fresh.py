#!/usr/bin/env python
"""Summarise a fresh-token sweep (scripts/sweep_fresh.py): per budget and arm, the best (LR, WD) by
final bpb, with the validation curve at fractions of the epoch and tokens-to-threshold."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

NAME = re.compile(r"^(?P<budget>[^_]+)_(?P<arm>.+)_lr(?P<lr>[^_]+)_wd(?P<wd>[^_]+)$")
FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)
THRESH = (2.5, 2.2, 2.0, 1.9, 1.8, 1.75)


def curve(run_dir: Path):
    pts = []
    for line in open(run_dir / "log.jsonl"):
        r = json.loads(line)
        if "val_bpb" in r:
            pts.append((r["tokens"], r["val_bpb"]))
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_root")
    p.add_argument("--csv", default="")
    a = p.parse_args()
    rows, groups = [], defaultdict(list)
    for res in sorted(Path(a.out_root).glob("*/results.json")):
        m = NAME.match(res.parent.name)
        if not m:
            continue
        r = json.loads(res.read_text())
        pts = curve(res.parent)
        total = r["tokens_total"]
        at = {}
        for f in FRACS:
            cand = [v for t, v in pts if t <= f * total + 1]
            at[f] = min(cand) if cand else None
        to = {}
        for th in THRESH:
            hit = [t for t, v in pts if v <= th]
            to[th] = hit[0] if hit else None
        row = {**m.groupdict(), "name": res.parent.name, "zero_shot": r["zero_shot_bpb"], "final": r["converged_bpb"],
               "tokens_M": total / 1e6, "stop": r.get("stop_reason"),
               **{f"at{int(f*100)}pct": at[f] for f in FRACS}, **{f"to{th}_M": (to[th] or 0) / 1e6 for th in THRESH}}
        rows.append(row)
        groups[(row["budget"], row["arm"])].append(row)
    print("| budget | arm | best lr | best wd | zero-shot | " + " | ".join(f"bpb @{int(f*100)}%" for f in FRACS)
          + " | final (after decay) | " + " | ".join(f"tokens to {th}" for th in THRESH) + " |")
    print("| --- " * (5 + len(FRACS) + 1 + len(THRESH)) + "|")
    for key in sorted(groups, key=lambda k: (-float(k[0].rstrip('B')), k[1])):
        b = min(groups[key], key=lambda x: x["final"])
        ats = " | ".join(f"{b[f'at{int(f*100)}pct']:.3f}" if b[f'at{int(f*100)}pct'] else "-" for f in FRACS)
        tos = " | ".join(f"{b[f'to{th}_M']:.0f}M" if b[f'to{th}_M'] else "-" for th in THRESH)
        print(f"| {key[0]} | {key[1]} | {b['lr']} | {b['wd']} | {b['zero_shot']:.3f} | {ats} | **{b['final']:.4f}** | {tos} |")
    print(f"\n{len(rows)} runs. Columns '@x%' = best validation bpb seen within the first x% of the epoch; "
          "'tokens to' = fresh tokens consumed when validation bpb first reached the threshold.")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
