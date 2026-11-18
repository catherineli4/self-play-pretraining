#!/usr/bin/env bash
# Fresh-token (single-epoch) pre-pretraining comparison at 1m, unattended:
#   data: 2B-byte DCLM training stream (first 200 MB of each of the 10 global shards) + a separate 20 MB
#         held-out validation stream from a different local shard; self-play checkpoints from GCS
#   runs: {1B, 2B} x {scratch std 0.02, scratch std 0.005, warm final, warm round-2800} x LR {3e-3,1e-2} x WD {0,0.1}
#         each exactly one pass over the first N bytes (no repetition), paper's 200-step cosine decay at the end
#   then: summary table, mirror to GCS (also every 10 min while running, for spot preemptions)
# Launch: setsid nohup bash scripts/run_fresh_epoch.sh > /mnt/localssd/spp-runs/run_fresh_epoch.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/catherine/miniconda3/envs/optim-study/bin/python}
D=${D:-/mnt/localssd/spp-data}; R=${R:-/mnt/localssd/spp-runs}; PP=${PP:-/mnt/localssd/spp-pp1m-fresh}
GCS=${GCS:-gs://cmu-gpucloud-catheri4/self-play-pretraining}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
export TRITON_CACHE_DIR=/mnt/localssd/triton-cache WANDB_CACHE_DIR=/mnt/localssd/wandb-cache WANDB_DATA_DIR=/mnt/localssd/wandb-data WANDB_CONFIG_DIR=/mnt/localssd/wandb-config
export HF_HOME=/mnt/localssd/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TMPDIR=/mnt/localssd/tmp
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p "$D" "$R/sp_1m_seed0" "$PP"

for r in 002800 008192; do [ -f "$R/sp_1m_seed0/models_round_$r.pt" ] || gcloud storage cp "$GCS/sp_1m_seed0/models_round_$r.pt" "$R/sp_1m_seed0/"; done
if [ ! -f "$D/dclm_val.bin" ]; then
  log "building dclm_val.bin (20 MB, global shard 1 / local shard 9)"
  "$PY" scripts/prepare_data.py dclm "$D" --name dclm_val.bin --mb_per_shard 20 --n_global 1 --local_shards 9 2>&1 | grep -v -i warn | tail -2
fi
if [ ! -f "$D/dclm2b.bin" ]; then
  log "building dclm2b.bin (200 MB from each of the 10 global shards, local shards 0,1,2)"
  "$PY" scripts/prepare_data.py dclm "$D" --name dclm2b.bin --mb_per_shard 200 --n_global 10 --local_shards 0,1,2 2>&1 | grep -v -i warn | tail -3
fi
ls -la "$D"
( while true; do sleep 600; gcloud storage rsync -r "$PP" "$GCS/pp1m-fresh" > /dev/null 2>&1; done ) & SYNC=$!

log "sweep start"
"$PY" scripts/sweep_fresh.py --corpus "$D/dclm2b.bin" --val_corpus "$D/dclm_val.bin" --out_root "$PP" --gpus "$GPUS" \
  --warm_final "$R/sp_1m_seed0/models_round_008192.pt" --warm_r2800 "$R/sp_1m_seed0/models_round_002800.pt" \
  --wandb_project self-play-prepretraining-1m-fresh > "$PP/sweep.out" 2>&1
log "sweep done (rc $?)"
"$PY" scripts/collect_fresh.py "$PP" --csv "$PP/results.csv" > "$PP/summary.md" 2>&1
kill $SYNC 2>/dev/null
gcloud storage rsync -r "$PP" "$GCS/pp1m-fresh" > /dev/null 2>&1
gcloud storage cp "$D/dclm2b.bin" "$D/dclm_val.bin" "$GCS/data/" > /dev/null 2>&1
log "DONE: summary at $PP/summary.md, mirrored to $GCS/pp1m-fresh/"
