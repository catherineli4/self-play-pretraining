#!/usr/bin/env bash
# Push offline W&B runs under $RUNS to the cloud every 5 minutes (use when training was launched with
# WANDB_MODE=offline, e.g. because no API key was configured at launch time).
#   wandb login            # or export WANDB_API_KEY=...
#   setsid nohup bash scripts/wandb_sync.sh > /mnt/localssd/spp-runs/wandb_sync.out 2>&1 < /dev/null &
RUNS=${RUNS:-/mnt/localssd/spp-runs}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/mnt/localssd/wandb-cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/mnt/localssd/wandb-config}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/mnt/localssd/wandb-data}
WANDB=${WANDB:-/home/catherine/miniconda3/envs/optim-study/bin/wandb}
while true; do
  for d in "$RUNS"/*/wandb/offline-run-*; do
    [[ -d "$d" ]] && "$WANDB" sync "$d" 2>&1 | tail -1
  done
  sleep ${INTERVAL:-300}
done
