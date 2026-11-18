#!/usr/bin/env bash
# Unattended 25m pipeline (paper App. A at the paper's model size), one seed:
#   1. wait for the 1m self-play run to finish, then start 25m self-play on GPUs 0-6 (paper budget: 34.4B tokens)
#   2. wait for the 1m uniform baseline to finish, then run stage 1 of the from-scratch init-scale sweep on GPU 7
#      (global std x LR x WD, 25m, Metamath)
#   3. when 25m self-play is done: stage 2 of the sweep (embedding / residual / head scales at the best std)
#      plus the warm-start arm from the final 25m checkpoint, on all 8 GPUs
#   4. write the summary table
# Launch:  setsid nohup bash scripts/orchestrate_25m.sh > /mnt/localssd/spp-runs/orchestrate_25m.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/catherine/miniconda3/envs/optim-study/bin/python}
RUNS=${RUNS:-/mnt/localssd/spp-runs}
DATA=${DATA:-/mnt/localssd/spp-data}
PP=${PP:-/mnt/localssd/spp-pp25m}
WANDB_PROJECT=${WANDB_PROJECT:-self-play-pretraining}
GCS_PATH=${GCS_PATH-gs://cmu-gpucloud-catheri4/self-play-pretraining}
export TRITON_CACHE_DIR=/mnt/localssd/triton-cache WANDB_CACHE_DIR=/mnt/localssd/wandb-cache WANDB_DATA_DIR=/mnt/localssd/wandb-data WANDB_CONFIG_DIR=/mnt/localssd/wandb-config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PP"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
wait_for_run() {  # wait until a train.py run with this out_dir has no live process
  while pgrep -f "scripts/train.py.*--out_dir $1( |$)" > /dev/null; do sleep 60; done
}

# ---- 1. 25m self-play after the 1m self-play ends
if [[ ! -f "$RUNS/sp_25m_seed0/ckpt_last.pt" || ! -f "$RUNS/sp_25m_seed0/models_round_006560.pt" ]]; then
  log "waiting for sp_1m_seed0 to finish"; wait_for_run "$RUNS/sp_1m_seed0"
  log "starting sp_25m_seed0 on GPUs 0-6"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 "$PY" -m torch.distributed.run --standalone --nproc_per_node 7 scripts/train.py \
    --config configs/replicate_25m.json --data selfplay --seed 0 --out_dir "$RUNS/sp_25m_seed0" \
    --eval metamath="$DATA/metamath.bin" --wandb_project "$WANDB_PROJECT" --wandb_name sp_25m_seed0 --gcs_path "$GCS_PATH" \
    > "$RUNS/sp_25m_seed0.out" 2>&1 &
  SP_PID=$!
else
  log "sp_25m_seed0 already complete"; SP_PID=""
fi

# ---- 2. stage-1 scratch sweep on GPU 7 after the uniform baseline ends
log "waiting for uniform_1m_seed0 to finish"; wait_for_run "$RUNS/uniform_1m_seed0"
log "stage 1: global init std sweep (25m scratch) on GPU 7"
"$PY" scripts/sweep_pretrain.py --corpus "$DATA/metamath.bin" --out_root "$PP" --gpus 7 --sizes 25m \
  --stds 0.005,0.01,0.02,0.03,0.05,0.1 --lrs 1e-3,3e-3,1e-2 --wds 0,0.1 --wandb_project "$WANDB_PROJECT-pp25m" \
  --extra "--micro_batch 16" > "$PP/sweep_stage1.out" 2>&1
log "stage 1 done"

# ---- 3. stage 2 + warm-start arm after 25m self-play ends
[[ -n "$SP_PID" ]] && { log "waiting for sp_25m_seed0"; wait "$SP_PID"; }
WARM=$(ls -1 "$RUNS"/sp_25m_seed0/models_round_*.pt 2>/dev/null | sort | tail -1)
BEST_STD=$("$PY" - <<PYEOF
import json, glob, re, collections
best = collections.defaultdict(lambda: 9e9)
for f in glob.glob("$PP/25m_scratch_std*/results.json"):
    r = json.load(open(f)); m = re.search(r"_std([0-9.]+)_", f)
    if r.get("converged_bpb") is not None: best[m.group(1)] = min(best[m.group(1)], r["converged_bpb"])
print(min(best, key=best.get) if best else "0.02")
PYEOF
)
log "stage 2: emb/resid/head scales at std=$BEST_STD plus warm start from $WARM, on all GPUs"
"$PY" scripts/sweep_pretrain.py --corpus "$DATA/metamath.bin" --out_root "$PP" --gpus 0,1,2,3,4,5,6,7 --sizes 25m \
  --stds "" --base_std "$BEST_STD" --emb_stds 0.1,0.3,1.0 --resid_scales 1,0 --head_stds 0 \
  --lrs 1e-3,3e-3,1e-2 --wds 0,0.1 --warm_start "25m=$WARM" --wandb_project "$WANDB_PROJECT-pp25m" \
  --extra "--micro_batch 16" > "$PP/sweep_stage2.out" 2>&1
log "stage 2 done"
"$PY" scripts/collect_pretrain.py "$PP" --csv "$PP/results.csv" > "$PP/summary.md" 2>&1
log "summary written to $PP/summary.md"
