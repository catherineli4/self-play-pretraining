#!/usr/bin/env python
"""Build byte-stream evaluation files following paper App. B (the ones that are freely downloadable).

  python scripts/prepare_data.py metamath  data/   # set.mm at the paper's pinned commit -> metamath.bin
  python scripts/prepare_data.py cifar10   data/   # CIFAR-10 test batch -> cifar10_chw.bin, cifar10_hwc.bin
  python scripts/prepare_data.py text FILE data/   # any UTF-8 text file -> text.bin (bytes as-is)
  python scripts/prepare_data.py dclm data/ [--name dclm.bin --mb_per_shard 27 --n_global 10 --local_shards 0]
                                                   # DCLM-Baseline-1.0 document texts, concatenated (needs zstd on
                                                   # PATH; set HF_TOKEN for rate limits; shards cached in HF_HOME)
"""
import io
import os
import pickle
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

DCLM_REPO = "mlfoundations/dclm-baseline-1.0"
SETMM = "https://raw.githubusercontent.com/metamath/set.mm/bcfef9892b6103ba9046bf683b4903d2ad081a41/set.mm"
CIFAR = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def metamath(out: Path):
    raw = urllib.request.urlopen(SETMM).read().decode("utf-8", "replace")
    raw = re.sub(r"\$\(.*?\$\)", "", raw, flags=re.S)       # drop comments
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    (out / "metamath.bin").write_bytes(("\n".join(lines) + "\n").encode("ascii", "replace"))


def cifar10(out: Path):
    import numpy as np
    buf = urllib.request.urlopen(CIFAR).read()
    with tarfile.open(fileobj=io.BytesIO(buf)) as tf:
        d = pickle.load(tf.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")
    x = d[b"data"].astype(np.uint8)                        # [10000, 3072] planar CHW
    (out / "cifar10_chw.bin").write_bytes(x.tobytes())
    hwc = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    (out / "cifar10_hwc.bin").write_bytes(np.ascontiguousarray(hwc).tobytes())


def dclm(out: Path, name: str = "dclm.bin", mb_per_shard: float = 27.0, n_global: int = 10, local_shards=(0,)):
    """Paper App. B: sample the ten global shards of DCLM-Baseline-1.0, keep only the document texts,
    concatenate (UTF-8 bytes). Takes mb_per_shard MB from each global shard, reading the given local
    shards' files in order until the quota is met."""
    import json
    import subprocess
    from huggingface_hub import hf_hub_download
    quota = int(mb_per_shard * 1e6)
    total, n_docs = 0, 0
    with open(out / name, "wb") as fo:
        for g in range(1, n_global + 1):
            written, done = 0, False
            for ls in local_shards:
                for k in range(100):
                    fname = f"global-shard_{g:02d}_of_10/local-shard_{ls}_of_10/shard_{k:08d}_processed.jsonl.zst"
                    try:
                        path = hf_hub_download(DCLM_REPO, fname, repo_type="dataset")
                    except Exception as e:  # EntryNotFound -> this local shard has no more files
                        if "404" in str(e) or "NotFound" in type(e).__name__:
                            break
                        raise
                    proc = subprocess.Popen(["zstd", "-dc", os.path.realpath(path)], stdout=subprocess.PIPE)   # HF cache uses symlinks
                    for line in proc.stdout:
                        text = json.loads(line).get("text", "")
                        if not text:
                            continue
                        b = (text + "\n").encode("utf-8")
                        fo.write(b)
                        written += len(b)
                        n_docs += 1
                        if written >= quota:
                            done = True
                            break
                    proc.kill()
                    proc.wait()
                    if done:
                        break
                if done:
                    break
            total += written
            print(f"global shard {g:02d}: {written/1e6:.1f} MB (last file {fname})", flush=True)
    print(f"{name}: {total/1e6:.1f} MB, {n_docs} documents", flush=True)

def main():
    which = sys.argv[1]
    if which == "text":
        src, out = Path(sys.argv[2]), Path(sys.argv[3])
        out.mkdir(parents=True, exist_ok=True)
        (out / "text.bin").write_bytes(src.read_bytes())
        return
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if which == "dclm":
        import argparse
        ap = argparse.ArgumentParser(prog="prepare_data.py dclm OUT_DIR")
        ap.add_argument("--name", default="dclm.bin")
        ap.add_argument("--mb_per_shard", type=float, default=27.0)
        ap.add_argument("--n_global", type=int, default=10)
        ap.add_argument("--local_shards", default="0", help="comma-separated local shard indices to read, in order")
        a = ap.parse_args(sys.argv[3:])
        dclm(out, a.name, a.mb_per_shard, a.n_global, tuple(int(x) for x in a.local_shards.split(",")))
        return
    {"metamath": metamath, "cifar10": cifar10}[which](out)


if __name__ == "__main__":
    main()
