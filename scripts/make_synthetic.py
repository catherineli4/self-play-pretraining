#!/usr/bin/env python
"""Write a synthetic pre-pretraining byte stream (see spp/synthetic.py) to a file, in chunks.

  python scripts/make_synthetic.py --kind kdyck --out data/synthetic/kdyck.bin --bytes 2e9 --seed 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spp.synthetic import GENERATORS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=sorted(GENERATORS))
    p.add_argument("--out", required=True)
    p.add_argument("--bytes", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=float, default=1e8, help="bytes per generated chunk")
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gen = GENERATORS[a.kind]
    total, t0, i = 0, time.time(), 0
    with open(out, "wb") as f:
        while total < a.bytes:
            n = int(min(a.chunk, a.bytes - total))
            kw = {"verbose": i == 0} if a.kind == "nca" else {}
            arr = gen(n, seed=a.seed * 100_003 + i, **kw)
            f.write(arr.tobytes())
            total += len(arr)
            i += 1
            print(f"{a.kind}: {total/1e6:.0f}M bytes, {time.time()-t0:.0f}s", flush=True)
    print(f"wrote {out} ({total/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
