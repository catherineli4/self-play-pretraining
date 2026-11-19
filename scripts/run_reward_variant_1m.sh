#!/usr/bin/env bash
# Reward variant at 1m: no AdamW step operator in the learning-progress reward
# (tangent = theta_past - theta_e instead of P_e * (theta_past - theta_e), --tangent_precond identity).
#   1. self-play on all 8 GPUs with configs/replicate_1m.json (same seed, rounds, pool as sp_1m_seed0),
#      checkpoints every 100 rounds mirrored to GCS, resumable (--resume auto; re-run this script after a preemption)
#   2. fresh-DCLM single-epoch warm-start comparison (1B/2B, LR {3e-3,1e-2} x WD {0,0.1}) of its round-2800 and
#      final checkpoints, written into the existing pp1m-fresh sweep dir so collect_fresh.py shows every arm
#      (scratch, warm_final, warm_r2800, warm_rawdelta_*) in one table
# Launch: setsid nohup bash scripts/run_reward_variant_1m.sh > /mnt/localssd/spp-runs/run_reward_variant_1m.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/catherine/miniconda3/envs/optim-study/bin/python}
D=${D:-/mnt/localssd/spp-data}; R=${R:-/mnt/localssd/spp-runs}; PP=${PP:-/mnt/localssd/spp-pp1m-fresh}
GCS=${GCS:-gs://cmu-gpucloud-catheri4/self-play-pretraining}
NAME=${NAME:-sp_1m_rawdelta_seed0}; SP="$R/$NAME"
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
export TRITON_CACHE_DIR=/mnt/localssd/triton-cache WANDB_CACHE_DIR=/mnt/localssd/wandb-cache WANDB_DATA_DIR=/mnt/localssd/wandb-data WANDB_CONFIG_DIR=/mnt/localssd/wandb-config
export HF_HOME=/mnt/localssd/hf TMPDIR=/mnt/localssd/tmp
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }
running() { pgrep -f "$1" > /dev/null; }
wait_gone() { while pgrep -f "$1" > /dev/null; do sleep 60; done; }
mkdir -p "$R" "$PP" "$D" "$TRITON_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_CONFIG_DIR" "$TMPDIR"
for f in metamath.bin dclm.bin dclm2b.bin dclm_val.bin; do [ -f "$D/$f" ] || gcloud storage cp "$GCS/data/$f" "$D/"; done

PAT="out_dir $SP "
if [ ! -f "$SP/models_round_008192.pt" ]; then
  if running "$PAT"; then
    log "attaching to the running self-play $NAME"
  else
    log "launching self-play $NAME on GPUs $GPUS ($NPROC procs), tangent_precond=identity"
    CUDA_VISIBLE_DEVICES=$GPUS setsid nohup "$PY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config configs/replicate_1m.json --data selfplay --seed 0 --out_dir "$SP" --resume auto --tangent_precond identity \
      --eval metamath="$D/metamath.bin" --eval dclm="$D/dclm.bin" \
      --wandb_project self-play-pretraining --wandb_name "$NAME" --gcs_path "$GCS" > "$R/$NAME.out" 2>&1 < /dev/null &
    sleep 10
  fi
  wait_gone "$PAT"
fi
if [ ! -f "$SP/models_round_008192.pt" ]; then
  log "self-play did not finish (no $SP/models_round_008192.pt); fix and re-run this script"; exit 1
fi
log "self-play done; fresh-DCLM warm-start sweep of the round-2800 and final checkpoints on GPUs $GPUS"
( while true; do sleep 600; gcloud storage rsync -r "$PP" "$GCS/pp1m-fresh" > /dev/null 2>&1; done ) & SYNC=$!
"$PY" scripts/sweep_fresh.py --corpus "$D/dclm2b.bin" --val_corpus "$D/dclm_val.bin" --out_root "$PP" --gpus "$GPUS" --scratch_stds "" \
  --arm "warm_rawdelta_final=$SP/models_round_008192.pt" --arm "warm_rawdelta_r2800=$SP/models_round_002800.pt" \
  --wandb_project self-play-prepretraining-1m-fresh > "$PP/sweep_rawdelta.out" 2>&1
log "sweep done (rc $?)"
"$PY" scripts/collect_fresh.py "$PP" --csv "$PP/results.csv" > "$PP/summary.md" 2>&1
kill $SYNC 2>/dev/null
gcloud storage rsync -r "$PP" "$GCS/pp1m-fresh" > /dev/null 2>&1
log "DONE: $PP/summary.md (mirrored to $GCS/pp1m-fresh/)"
