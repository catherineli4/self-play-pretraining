#!/usr/bin/env python
"""Zero-shot bits-per-byte and ICL evaluation of a learner checkpoint."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spp.data import (arithmetic_records, evaluate_bpb, load_byte_file, random_bytes_records,  # noqa: E402
                      records_from_bytes)
from spp.icl import evaluate_icl  # noqa: E402
from spp.train import load_learner  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--eval", action="append", default=[], help="name=path (raw bytes)")
    p.add_argument("--ctx", type=int, default=256, help="record context (paper: 256 for App. B datasets)")
    p.add_argument("--records", type=int, default=1024)
    p.add_argument("--icl", action="store_true")
    p.add_argument("--icl_instances", type=int, default=256)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    model, state = load_learner(a.ckpt, a.device)
    L = a.ctx - 1
    out = {"round": state["round"], "tokens": state["tokens_seen"]}
    out["random_bytes_bpb"] = evaluate_bpb(model, random_bytes_records(a.records, L), device=a.device)
    out["arithmetic_bpb"] = evaluate_bpb(model, arithmetic_records(a.records, L), device=a.device)
    for item in a.eval:
        name, path = item.split("=", 1)
        recs = records_from_bytes(load_byte_file(path), L, max_records=a.records)
        out[f"{name}_bpb"] = evaluate_bpb(model, recs, device=a.device)
    if a.icl:
        out["icl"] = evaluate_icl(model, n_instances=a.icl_instances, ctx=model.cfg.max_seq_len, device=a.device)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
