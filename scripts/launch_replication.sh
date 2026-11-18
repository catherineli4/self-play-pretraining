#!/usr/bin/env bash
# Launch the 1M-parameter self-play replication (multi-GPU) + the uniform-prior baseline, log to W&B,
# mirror checkpoints to GCS, and serve the dashboard. Everything is started with setsid+nohup, so it
# survives closing the terminal / SSH session.
#
#   wandb login                      # once; or export WANDB_API_KEY=...   (WANDB_MODE=offline also works)
#   bash scripts/launch_replication.sh
#
# Env knobs: GPUS_SP (comma list for self-play, default 0-6), GPU_UNIFORM (one GPU, empty = skip baseline),
#            RUNS (output root), DATA (eval byte files dir), WANDB_PROJECT, WANDB_MODE, GCS_PATH (empty = no upload),
#            SEED, PORT, CFG, SPP_PYLIBS (dir holding jvp_flash_attention if not installed in the env)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/catherine/miniconda3/envs/optim-study/bin/python}
RUNS=${RUNS:-/mnt/localssd/spp-runs}
DATA=${DATA:-/mnt/localssd/spp-data}
WANDB_PROJECT=${WANDB_PROJECT:-self-play-pretraining}
SEED=${SEED:-0}
GPUS_SP=${GPUS_SP:-0,1,2,3,4,5,6}
GPU_UNIFORM=${GPU_UNIFORM-7}
PORT=${PORT:-8787}
CFG=${CFG:-configs/replicate_1m.json}
GCS_PATH=${GCS_PATH-gs://cmu-gpucloud-catheri4/self-play-pretraining}
SCRATCH=${SCRATCH:-/mnt/localssd}
# caches off the (full) home disk
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/triton-cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$SCRATCH/wandb-cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-$SCRATCH/wandb-data}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$SCRATCH/wandb-config}
mkdir -p "$RUNS" "$TRITON_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_CONFIG_DIR"
EVALS=(--eval metamath="$DATA/metamath.bin")
NPROC=$(echo "$GPUS_SP" | tr ',' '\n' | grep -c .)

NAME=sp_1m_seed$SEED
CUDA_VISIBLE_DEVICES=$GPUS_SP setsid nohup "$PY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  scripts/train.py --config "$CFG" --data selfplay --seed "$SEED" \
  --out_dir "$RUNS/$NAME" "${EVALS[@]}" --wandb_project "$WANDB_PROJECT" --wandb_name "$NAME" --gcs_path "$GCS_PATH" \
  > "$RUNS/$NAME.out" 2>&1 < /dev/null &
echo "self-play      -> GPUs $GPUS_SP ($NPROC procs)  $RUNS/$NAME  (pid $!)"

if [[ -n "$GPU_UNIFORM" ]]; then
  NAME=uniform_1m_seed$SEED
  CUDA_VISIBLE_DEVICES=$GPU_UNIFORM setsid nohup "$PY" scripts/train.py --config "$CFG" --data uniform --seed "$SEED" \
    --out_dir "$RUNS/$NAME" "${EVALS[@]}" --wandb_project "$WANDB_PROJECT" --wandb_name "$NAME" --gcs_path "$GCS_PATH" \
    > "$RUNS/$NAME.out" 2>&1 < /dev/null &
  echo "uniform prior  -> GPU $GPU_UNIFORM  $RUNS/$NAME  (pid $!)"
fi

if ! pgrep -f "scripts/dashboard.py" > /dev/null; then
  setsid nohup "$PY" scripts/dashboard.py --runs "$RUNS" --port "$PORT" > "$RUNS/dashboard.out" 2>&1 < /dev/null &
  echo "dashboard      -> http://localhost:$PORT  (pid $!)   remote: ssh -L $PORT:localhost:$PORT <this host>"
else
  echo "dashboard      -> already running on http://localhost:$PORT"
fi
echo "wandb          -> project '$WANDB_PROJECT' (mode: ${WANDB_MODE:-online})"
echo "gcs            -> ${GCS_PATH:-(disabled)}/<run>/  (ckpt_last.pt = resumable; models_round_*.pt = learner + generator weights)"
echo "stop all       -> pkill -f scripts/train.py; pkill -f scripts/dashboard.py"
