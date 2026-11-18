#!/usr/bin/env bash
# 25m pipeline, idempotent and re-entrant: rerun it any time (after a preemption: scripts/bootstrap_instance.sh
# first). Steps already finished are skipped; steps already running (self-play, a pre-pretraining run) are attached to.
#  A. self-play 25m on GPUs 0-6 (paper budget: 6560 rounds = 34.4B synthetic tokens), checkpoint + GCS mirror
#     every 50 rounds, resumes from ckpt_last.pt (lookback rebuilt from the mirrored snapshots)
#  B. GPU 7, meanwhile: synthetic curricula from "Rethinking Pre-pretraining" (k-Dyck, NCA, DeclRef; 2B bytes
#     each + 20 MB held-out), 25m pre-pretraining on each (one pass over 2B tokens, exported at 1B and 2B),
#     then the fresh-DCLM downstream comparison (one epoch over the first 1B and 2B bytes) for: scratch
#     (std 0.02, 0.005), self-play at 1B and 2B synthetic tokens (rounds 200, 400), each curriculum at 2B
#  C. when self-play finishes, the remaining downstream runs plus the final self-play checkpoint move to all
#     8 GPUs; then summary and mirror to GCS
# Launch: setsid nohup bash scripts/run_25m_pipeline.sh > /mnt/localssd/spp-runs/run_25m_pipeline.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/catherine/miniconda3/envs/optim-study/bin/python}
D=${D:-/mnt/localssd/spp-data}; SYN=$D/synthetic; R=${R:-/mnt/localssd/spp-runs}; SP=$R/sp_25m_seed0
PP=${PP:-/mnt/localssd/spp-pp25m}
GCS=${GCS:-gs://cmu-gpucloud-catheri4/self-play-pretraining}
SP_ROUNDS=6560
export TRITON_CACHE_DIR=/mnt/localssd/triton-cache WANDB_CACHE_DIR=/mnt/localssd/wandb-cache WANDB_DATA_DIR=/mnt/localssd/wandb-data WANDB_CONFIG_DIR=/mnt/localssd/wandb-config
export HF_HOME=/mnt/localssd/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TMPDIR=/mnt/localssd/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
running() { pgrep -f "$1" > /dev/null; }
wait_gone() { while pgrep -f "$1" > /dev/null; do sleep 60; done; }
mkdir -p "$D" "$SYN" "$R" "$SP" "$PP/ppt"
for f in metamath.bin dclm.bin dclm2b.bin dclm_val.bin; do [ -f "$D/$f" ] || gcloud storage cp "$GCS/data/$f" "$D/"; done
while running "out_root /mnt/localssd/spp-pp1m-fresh"; do sleep 30; done

# ---- A. self-play 25m (GPUs 0-6)
SP_PAT="scripts/train.py.*--out_dir $SP( |$)"
if [ -f "$SP/models_round_00${SP_ROUNDS}.pt" ]; then
  log "sp_25m_seed0 already complete"
elif running "$SP_PAT"; then
  log "sp_25m_seed0 already running; attaching"
else
  if [ ! -f "$SP/ckpt_last.pt" ] && gcloud storage ls "$GCS/sp_25m_seed0/ckpt_last.pt" > /dev/null 2>&1; then
    log "restoring sp_25m_seed0 from GCS"; gcloud storage cp "$GCS/sp_25m_seed0/*" "$SP/" > /dev/null 2>&1
  fi
  log "launching sp_25m_seed0 on GPUs 0-6 (resume: $([ -f "$SP/ckpt_last.pt" ] && echo yes || echo no))"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PY" -m torch.distributed.run --standalone --nproc_per_node 7 scripts/train.py \
    --config configs/replicate_25m.json --data selfplay --seed 0 --out_dir "$SP" --resume auto \
    --eval metamath="$D/metamath.bin" --eval dclm="$D/dclm.bin" \
    --wandb_project self-play-pretraining --wandb_name sp_25m_seed0 --gcs_path "$GCS" >> "$R/sp_25m_seed0.out" 2>&1 &
  sleep 5
fi
( while true; do sleep 600; gcloud storage rsync -r "$PP" "$GCS/pp25m" > /dev/null 2>&1; done ) & SYNC=$!

# ---- B. GPU 7: synthetic curricula -> pre-pretraining -> downstream stage 1
for kind in kdyck nca declref; do
  wait_gone "make_synthetic.py --kind $kind "
  if [ ! -f "$SYN/$kind.bin" ]; then
    log "generating $kind (2B + 20 MB held-out)"
    CUDA_VISIBLE_DEVICES=7 "$PY" scripts/make_synthetic.py --kind $kind --out "$SYN/${kind}_val.bin" --bytes 2e7 --seed 1 > "$SYN/${kind}_val.log" 2>&1
    CUDA_VISIBLE_DEVICES=7 "$PY" scripts/make_synthetic.py --kind $kind --out "$SYN/$kind.bin" --bytes 2e9 --seed 0 > "$SYN/$kind.log" 2>&1 || log "$kind generation FAILED"
    ( gcloud storage cp "$SYN/$kind.bin" "$SYN/${kind}_val.bin" "$GCS/data/synthetic/" > /dev/null 2>&1 ) &
  fi
done
for kind in kdyck nca declref; do
  PAT="pretrain_natural.py.*--out_dir $PP/ppt/$kind( |$)"
  if running "$PAT"; then log "pre-pretraining on $kind already running; waiting"; wait_gone "$PAT"; fi
  if [ ! -f "$PP/ppt/$kind/ckpt_final.pt" ]; then
    log "pre-pretraining 25m on $kind (2B tokens, one pass)"
    CUDA_VISIBLE_DEVICES=7 "$PY" scripts/pretrain_natural.py --corpus "$SYN/$kind.bin" --val_corpus "$SYN/${kind}_val.bin" \
      --out_dir "$PP/ppt/$kind" --preset 25m --micro_batch 16 --max_epochs 1 --lr 1e-3 --weight_decay 0.1 \
      --save_at_tokens 1e9 --save_final 1 --wandb_project self-play-pretraining-25m-ppt --wandb_name "ppt_$kind" \
      > "$PP/ppt/$kind.out" 2>&1 || log "ppt $kind FAILED"
  fi
done
log "waiting for self-play checkpoints at rounds 200 and 400 (1B and 2B synthetic tokens)"
until [ -f "$SP/models_round_000200.pt" ] && [ -f "$SP/models_round_000400.pt" ]; do sleep 120; done
ARMS=(--corpus "$D/dclm2b.bin" --val_corpus "$D/dclm_val.bin" --out_root "$PP" --preset 25m --scratch_stds 0.02,0.005
      --arm "sp_1B=$SP/models_round_000200.pt" --arm "sp_2B=$SP/models_round_000400.pt"
      --arm "kdyck_2B=$PP/ppt/kdyck/ckpt_final.pt" --arm "nca_2B=$PP/ppt/nca/ckpt_final.pt" --arm "declref_2B=$PP/ppt/declref/ckpt_final.pt"
      --lrs 1e-3,3e-3 --wds 0.1 --extra "--micro_batch 16" --wandb_project self-play-prepretraining-25m-fresh)
if running "$SP_PAT"; then
  log "downstream stage 1 on GPU 7 while self-play runs"
  "$PY" scripts/sweep_fresh.py "${ARMS[@]}" --gpus 7 > "$PP/sweep_stage1_gpu7.out" 2>&1 & S1=$!
  while kill -0 "$S1" 2>/dev/null && running "$SP_PAT"; do sleep 60; done
  if kill -0 "$S1" 2>/dev/null; then
    log "self-play finished: moving the remaining downstream runs to all GPUs (restarting the one in progress)"
    pkill -f "pretrain_natural.py.*--out_dir $PP/(1B|2B)_"; kill "$S1" 2>/dev/null; wait "$S1" 2>/dev/null
    for d in "$PP"/1B_* "$PP"/2B_*; do [ -d "$d" ] && [ ! -f "$d/results.json" ] && rm -rf "$d"; done
  fi
fi

# ---- C. everything remaining, plus the final self-play checkpoint, on all GPUs
wait_gone "$SP_PAT"
FINAL=$(ls -1 "$SP"/models_round_*.pt | sort | tail -1)
log "downstream on all GPUs (remaining stage-1 runs + final self-play checkpoint $FINAL)"
"$PY" scripts/sweep_fresh.py "${ARMS[@]}" --arm "sp_final=$FINAL" --gpus 0,1,2,3,4,5,6,7 > "$PP/sweep_all.out" 2>&1
"$PY" scripts/collect_fresh.py "$PP" --csv "$PP/results.csv" > "$PP/summary.md" 2>&1
kill "$SYNC" 2>/dev/null
gcloud storage rsync -r "$PP" "$GCS/pp25m" > /dev/null 2>&1
log "DONE: $PP/summary.md (mirrored to $GCS/pp25m/)"
