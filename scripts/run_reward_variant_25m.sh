#!/usr/bin/env bash
# Reward variant at 25m: no AdamW step operator in the learning-progress reward (--tangent_precond identity),
# full length (6560 rounds, as sp_25m_seed0), on all 8 GPUs, then the fresh-DCLM downstream comparison of its
# round-200 / round-400 / final checkpoints (the same arms as the baseline's sp_1B / sp_2B / sp_final) into the
# existing pp25m sweep dir so collect_fresh.py reports every 25m arm in one table.
# Re-entrant and resumable (--resume auto); after a spot preemption run scripts/bootstrap_instance.sh, then this again.
# Launch: setsid nohup bash scripts/run_reward_variant_25m.sh > /mnt/localssd/spp-runs/run_reward_variant_25m.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/catherine/miniconda3/envs/optim-study/bin/python}
D=${D:-/mnt/localssd/spp-data}; R=${R:-/mnt/localssd/spp-runs}; PP=${PP:-/mnt/localssd/spp-pp25m}
GCS=${GCS:-gs://cmu-gpucloud-catheri4/self-play-pretraining}
NAME=${NAME:-sp_25m_rawdelta_seed0}; SP="$R/$NAME"; ROUNDS=${ROUNDS:-6560}
FINAL="$SP/models_round_$(printf '%06d' "$ROUNDS").pt"
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
export TRITON_CACHE_DIR=/mnt/localssd/triton-cache WANDB_CACHE_DIR=/mnt/localssd/wandb-cache WANDB_DATA_DIR=/mnt/localssd/wandb-data WANDB_CONFIG_DIR=/mnt/localssd/wandb-config
export HF_HOME=/mnt/localssd/hf TMPDIR=/mnt/localssd/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }
running() { pgrep -f "$1" > /dev/null; }
wait_gone() { while pgrep -f "$1" > /dev/null; do sleep 60; done; }
mkdir -p "$R" "$PP" "$D" "$TRITON_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_CONFIG_DIR" "$TMPDIR"
for f in metamath.bin dclm.bin dclm2b.bin dclm_val.bin; do [ -f "$D/$f" ] || gcloud storage cp "$GCS/data/$f" "$D/"; done

PAT="out_dir $SP "
if [ ! -f "$FINAL" ]; then
  if running "$PAT"; then
    log "attaching to the running self-play $NAME"
  else
    log "launching self-play $NAME on GPUs $GPUS ($NPROC procs), tangent_precond=identity, rounds=$ROUNDS"
    CUDA_VISIBLE_DEVICES=$GPUS setsid nohup "$PY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" scripts/train.py \
      --config configs/replicate_25m.json --rounds "$ROUNDS" --data selfplay --seed 0 --out_dir "$SP" --resume auto --tangent_precond identity \
      --eval metamath="$D/metamath.bin" --eval dclm="$D/dclm.bin" \
      --wandb_project self-play-pretraining --wandb_name "$NAME" --gcs_path "$GCS" >> "$R/$NAME.out" 2>&1 < /dev/null &
    sleep 10
  fi
  wait_gone "$PAT"
fi
if [ ! -f "$FINAL" ]; then
  log "self-play did not finish (no $FINAL); fix and re-run this script"; exit 1
fi
log "self-play done; fresh-DCLM warm-start sweep (r200 / r400 / final) on GPUs $GPUS"
( while true; do sleep 600; gcloud storage rsync -r "$PP" "$GCS/pp25m" > /dev/null 2>&1; done ) & SYNC=$!
"$PY" scripts/sweep_fresh.py --corpus "$D/dclm2b.bin" --val_corpus "$D/dclm_val.bin" --out_root "$PP" --gpus "$GPUS" --preset 25m --scratch_stds "" \
  --arm "sp_rawdelta_1B=$SP/models_round_000200.pt" --arm "sp_rawdelta_2B=$SP/models_round_000400.pt" --arm "sp_rawdelta_final=$FINAL" \
  --lrs 1e-3,3e-3 --wds 0.1 --extra "--micro_batch 16" --wandb_project self-play-prepretraining-25m-fresh > "$PP/sweep_rawdelta.out" 2>&1
log "sweep done (rc $?)"
"$PY" scripts/collect_fresh.py "$PP" --csv "$PP/results.csv" --thresholds 2.0,1.7,1.6,1.5,1.45,1.4 > "$PP/summary.md" 2>&1
kill $SYNC 2>/dev/null
gcloud storage rsync -r "$PP" "$GCS/pp25m" > /dev/null 2>&1
log "DONE: $PP/summary.md (mirrored to $GCS/pp25m/)"
