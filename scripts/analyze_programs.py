#!/usr/bin/env python
"""Count mathematical sequence families (App. D) in the pools dumped during training."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spp.structure import FAMILIES  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    a = p.parse_args()
    earliest, examples = {}, {}
    per_round = defaultdict(Counter)
    for f in sorted((Path(a.run_dir) / "programs").glob("round_*.jsonl")):
        e = int(f.stem.split("_")[1])
        for line in f.read_text().splitlines():
            r = json.loads(line)
            fam = r.get("family")
            if fam:
                per_round[e][fam] += 1
                if fam not in earliest:
                    earliest[fam], examples[fam] = e, r
    print("round  " + "  ".join(f"{x:>10}" for x in FAMILIES))
    for e in sorted(per_round):
        print(f"{e:5d}  " + "  ".join(f"{per_round[e][x]:10d}" for x in FAMILIES))
    print("\nearliest round per family:")
    for fam in FAMILIES:
        if fam in earliest:
            ex = examples[fam]
            print(f"  {fam:11s} round {earliest[fam]:6d}  S{ex['body']}F  -> {ex['output_head'][:10]}")
        else:
            print(f"  {fam:11s} not found")


if __name__ == "__main__":
    main()
