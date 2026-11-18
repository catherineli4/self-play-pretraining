#!/usr/bin/env bash
# Compute-matched from-scratch baselines for the 25m downstream comparison. The self-play learner consumed
# 1.05B (round 200), 2.10B (round 400) and 34.4B (round 6560) synthetic tokens before its downstream pass over
# 1B or 2B fresh DCLM tokens, so the matched baselines train from scratch on the same TOTAL number of fresh
# DCLM tokens, one pass, no repetition:
#   3.05B  = sp@1B + 2B          4.10B = sp@2B + 2B (also matches the 2B-token synthetic curricula + 2B)
#   36.4B  = sp final + 2B       (the 1B-downstream variants differ by < 3%; 2.05B ~ the existing 2B run)
# Data: a 36.4 GB DCLM stream (local shard 0 of each global shard; validation = local shard 9 of global shard 1,
# disjoint). The two small runs use single GPUs (lr 1e-3 and 3e-3); the 36.4B run uses 4 GPUs with DDP (lr 1e-3).
# Idempotent; runs after the 25m pipeline has finished. Results join /mnt/localssd/spp-pp25m (summary.md).
# Launch: setsid nohup bash scripts/run_matched_compute.sh > /mnt/localssd/spp-runs/run_matched_compute.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/catherine/miniconda3/envs/optim-study/bin/python}
D=${D:-/mnt/localssd/spp-data}; PP=${PP:-/mnt/localssd/spp-pp25m}
GCS=${GCS:-gs://cmu-gpucloud-catheri4/self-play-pretraining}
export TRITON_CACHE_DIR=/mnt/localssd/triton-cache WANDB_CACHE_DIR=/mnt/localssd/wandb-cache WANDB_DATA_DIR=/mnt/localssd/wandb-data WANDB_CONFIG_DIR=/mnt/localssd/wandb-config
export HF_HOME=/mnt/localssd/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TMPDIR=/mnt/localssd/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
WANDB=self-play-prepretraining-25m-fresh

while pgrep -f "prepare_data.py dclm.*dclm36b" > /dev/null; do sleep 60; done
if [ ! -f "$D/dclm36b.bin" ] || [ "$(stat -c %s "$D/dclm36b.bin")" -lt 36400000000 ]; then
  log "building dclm36b.bin"
  "$PY" scripts/prepare_data.py dclm "$D" --name dclm36b.bin --mb_per_shard 3640 --n_global 10 --local_shards 0 2>&1 | grep -v -i warn | tail -2
fi
log "waiting for the 25m pipeline to finish"
while pgrep -f "^bash scripts/run_25m_pipeline.sh" > /dev/null; do sleep 60; done
( while true; do sleep 600; gcloud storage rsync -r "$PP" "$GCS/pp25m" > /dev/null 2>&1; done ) & SYNC=$!

log "small matched runs (3.05B, 4.10B fresh tokens; lr 1e-3 and 3e-3) on GPUs 0-3"
"$PY" scripts/sweep_fresh.py --corpus "$D/dclm36b.bin" --val_corpus "$D/dclm_val.bin" --out_root "$PP" --gpus 0,1,2,3 --preset 25m \
  --budgets "3.05B=3.05e9,4.1B=4.1e9" --scratch_stds 0.02 --lrs 1e-3,3e-3 --wds 0.1 --extra "--micro_batch 16" \
  --wandb_project "$WANDB" > "$PP/sweep_matched_small.out" 2>&1 &
SMALL=$!

BIG="$PP/36.4B_scratch_std0.02_lr0.001_wd0.1"
if [ ! -f "$BIG/results.json" ]; then
  log "36.4B matched run on GPUs 4-7 (DDP)"
  rm -rf "$BIG"
  CUDA_VISIBLE_DEVICES=4,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc_per_node 4 scripts/pretrain_natural.py \
    --corpus "$D/dclm36b.bin" --corpus_tokens 36.4e9 --val_corpus "$D/dclm_val.bin" --max_epochs 1 --out_dir "$BIG" \
    --preset 25m --micro_batch 16 --lr 1e-3 --weight_decay 0.1 --init_std 0.02 --save_at_tokens 3.05e9,4.1e9,8e9,16e9 \
    --wandb_project "$WANDB" --wandb_name 36.4B_scratch_std0.02_lr0.001_wd0.1 --wandb_group 36.4B_scratch_std0.02 \
    > "$BIG.out" 2>&1 || log "36.4B run FAILED (see $BIG.out)"
fi
wait "$SMALL"
"$PY" scripts/collect_fresh.py "$PP" --csv "$PP/results.csv" > "$PP/summary.md" 2>&1
kill "$SYNC" 2>/dev/null
gcloud storage rsync -r "$PP" "$GCS/pp25m" > /dev/null 2>&1
gcloud storage cp "$D/dclm36b.bin" "$GCS/data/" > /dev/null 2>&1
log "DONE: $PP/summary.md (mirrored to $GCS/pp25m/)"
