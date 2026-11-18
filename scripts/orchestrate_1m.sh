#!/usr/bin/env bash
# Unattended 1m pre-pretraining pipeline (paper App. A at the 1m scale), one seed, DCLM text (270 MB stream built
# by scripts/prepare_data.py dclm), paper stopping rule (plateau) with a 1.5B-token safety cap, then the 200-step cosine decay:
#   stage 1 (now, all GPUs, shared with the running self-play jobs): from-scratch global init std sweep
#   stage 2 (after the 1m self-play finishes, GPUs free): embedding / residual / head scales at the best std
#   stage 3: warm-start arm from the final 1m self-play checkpoint, then the summary table.
# Launch:  setsid nohup bash scripts/orchestrate_1m.sh > /mnt/localssd/spp-runs/orchestrate_1m.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/catherine/miniconda3/envs/optim-study/bin/python}
RUNS=${RUNS:-/mnt/localssd/spp-runs}
DATA=${DATA:-/mnt/localssd/spp-data}
CORPUS=${CORPUS:-$DATA/dclm.bin}
PP=${PP:-/mnt/localssd/spp-pp1m}
WANDB_PROJECT=${WANDB_PROJECT:-self-play-prepretraining-1m-dclm}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
LRS=${LRS:-1e-3,3e-3,1e-2}
WDS=${WDS:-0,0.1}
EXTRA=${EXTRA:---max_tokens 1.5e9}
export TRITON_CACHE_DIR=/mnt/localssd/triton-cache WANDB_CACHE_DIR=/mnt/localssd/wandb-cache WANDB_DATA_DIR=/mnt/localssd/wandb-data WANDB_CONFIG_DIR=/mnt/localssd/wandb-config
mkdir -p "$PP"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
wait_for_run() { while pgrep -f "scripts/train.py.*--out_dir $1( |$)" > /dev/null; do sleep 60; done; }

log "stage 1: global init std sweep (1m scratch) on GPUs $GPUS, budget: $EXTRA"
"$PY" scripts/sweep_pretrain.py --corpus "$CORPUS" --out_root "$PP" --gpus "$GPUS" --sizes 1m \
  --stds 0.005,0.01,0.02,0.03,0.05,0.1 --lrs "$LRS" --wds "$WDS" --wandb_project "$WANDB_PROJECT" --extra "$EXTRA" \
  > "$PP/sweep_stage1.out" 2>&1
log "stage 1 done"

BEST_STD=$("$PY" - <<PYEOF
import json, glob, re, collections
best = collections.defaultdict(lambda: 9e9)
for f in glob.glob("$PP/1m_scratch_std*/results.json"):
    r = json.load(open(f)); m = re.search(r"_std([0-9.]+)_", f)
    if r.get("converged_bpb") is not None: best[m.group(1)] = min(best[m.group(1)], r["converged_bpb"])
print(min(best, key=best.get) if best else "0.02")
PYEOF
)
log "best std from stage 1: $BEST_STD; waiting for sp_1m_seed0 to finish before stage 2"; wait_for_run "$RUNS/sp_1m_seed0"
log "stage 2: embedding / residual / head scales at std=$BEST_STD"
"$PY" scripts/sweep_pretrain.py --corpus "$CORPUS" --out_root "$PP" --gpus "$GPUS" --sizes 1m \
  --stds "" --base_std "$BEST_STD" --emb_stds 0.1,0.3,1.0 --resid_scales 1,0 --head_stds 0 \
  --lrs "$LRS" --wds "$WDS" --wandb_project "$WANDB_PROJECT" --extra "$EXTRA" > "$PP/sweep_stage2.out" 2>&1
log "stage 2 done"

WARM=$(ls -1 "$RUNS"/sp_1m_seed0/models_round_*.pt 2>/dev/null | sort | tail -1)
log "stage 3: warm-start arm from $WARM"
"$PY" scripts/sweep_pretrain.py --corpus "$CORPUS" --out_root "$PP" --gpus "$GPUS" --sizes 1m \
  --stds "" --lrs "$LRS" --wds "$WDS" --warm_start "1m=$WARM" --wandb_project "$WANDB_PROJECT" --extra "$EXTRA" \
  > "$PP/sweep_stage3.out" 2>&1
log "stage 3 done"
"$PY" scripts/collect_pretrain.py "$PP" --csv "$PP/results.csv" > "$PP/summary.md" 2>&1
log "summary written to $PP/summary.md"
