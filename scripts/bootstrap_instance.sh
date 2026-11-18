#!/usr/bin/env bash
# Recover a fresh or preempted spot VM for this project. Idempotent; safe to rerun.
#   1. local-SSD RAID0 at /mnt/localssd (user's ~/optimizer-study-gcp/setup_localssd.sh)
#   2. NVIDIA kernel module for the running kernel (dkms) -> nvidia-smi
#   3. Miniconda + env optim-study (python 3.11, torch 2.8 cu128, numpy, wandb, huggingface_hub, jvp_flash_attention)
#   4. data and self-play checkpoints back from GCS
# Usage: setsid nohup bash scripts/bootstrap_instance.sh > ~/bootstrap_instance.out 2>&1 < /dev/null &
set -uo pipefail
GCS=${GCS:-gs://cmu-gpucloud-catheri4/self-play-pretraining}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# 1. SSD
mountpoint -q /mnt/localssd || bash ~/optimizer-study-gcp/setup_localssd.sh
sudo mkdir -p /mnt/localssd/tmp && sudo chmod 1777 /mnt/localssd/tmp
mkdir -p /mnt/localssd/{hf,spp-runs,spp-data,triton-cache,wandb-cache,wandb-data,wandb-config,tmp/pip-cache}
export TMPDIR=/mnt/localssd/tmp PIP_CACHE_DIR=/mnt/localssd/tmp/pip-cache

# 4 (early, needs no env). data + checkpoints from GCS
( D=/mnt/localssd/spp-data; R=/mnt/localssd/spp-runs/sp_1m_seed0; mkdir -p "$D" "$R"
  for f in metamath.bin dclm.bin; do [ -f "$D/$f" ] || gcloud storage cp "$GCS/data/$f" "$D/" >/dev/null 2>&1; done
  for r in 002800 008192; do [ -f "$R/models_round_$r.pt" ] || gcloud storage cp "$GCS/sp_1m_seed0/models_round_$r.pt" "$R/" >/dev/null 2>&1; done
  log "GCS data/checkpoints: $(ls $D $R | tr '\n' ' ')" ) &
GCS_PID=$!

# 2. NVIDIA module
( if ! nvidia-smi -L >/dev/null 2>&1; then
    K=$(uname -r); log "building NVIDIA module for kernel $K"
    while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "linux-headers-$K" > /mnt/localssd/tmp/apt-headers.log 2>&1 || log "apt headers FAILED: $(tail -2 /mnt/localssd/tmp/apt-headers.log)"
    sudo dkms autoinstall -k "$K" > /mnt/localssd/tmp/dkms.log 2>&1; tail -2 /mnt/localssd/tmp/dkms.log
    sudo modprobe nvidia && sudo modprobe nvidia_uvm
    sudo systemctl restart nvidia-persistenced 2>/dev/null || true
  fi
  nvidia-smi -L > /mnt/localssd/tmp/nvidia-smi.log 2>&1 && log "GPUs: $(wc -l < /mnt/localssd/tmp/nvidia-smi.log)" || log "nvidia-smi FAILED: $(head -2 /mnt/localssd/tmp/nvidia-smi.log)" ) &
NV_PID=$!

# 3. conda env
if [ ! -x /mnt/localssd/miniconda3/bin/conda ]; then
  log "installing Miniconda"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /mnt/localssd/tmp/miniconda.sh
  bash /mnt/localssd/tmp/miniconda.sh -b -p /mnt/localssd/miniconda3 > /mnt/localssd/tmp/miniconda-install.log 2>&1
fi
[ -e "$HOME/miniconda3" ] || ln -s /mnt/localssd/miniconda3 "$HOME/miniconda3"
source /mnt/localssd/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q "^optim-study " || { log "creating env optim-study"; conda create -y -q -n optim-study -c conda-forge --override-channels python=3.11 > /mnt/localssd/tmp/conda-create.log 2>&1; }
conda activate optim-study
PIP="pip install -q --cache-dir /mnt/localssd/tmp/pip-cache"
python -c "import huggingface_hub, numpy, wandb" 2>/dev/null || { log "pip: numpy wandb huggingface_hub"; $PIP numpy wandb huggingface_hub > /mnt/localssd/tmp/pip-base.log 2>&1 || log "pip base FAILED: $(tail -2 /mnt/localssd/tmp/pip-base.log)"; }
python -c "import torch; assert torch.__version__.startswith('2.8')" 2>/dev/null || { log "pip: torch 2.8 cu128 (large download)"; $PIP torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 > /mnt/localssd/tmp/pip-torch.log 2>&1 || log "pip torch FAILED: $(tail -2 /mnt/localssd/tmp/pip-torch.log)"; }
python -c "import jvp_flash_attention" 2>/dev/null || { $PIP jvp_flash_attention > /mnt/localssd/tmp/pip-jvp.log 2>&1 || log "jvp_flash_attention not installed (only needed for self-play reward speed)"; }
log "env: $(python -c 'import torch, numpy, wandb, huggingface_hub; print("torch", torch.__version__, "cuda build", torch.version.cuda)' 2>&1 | tail -1)"

wait $GCS_PID $NV_PID
python -c "import torch; print('torch sees', torch.cuda.device_count(), 'GPUs')" 2>&1 | tail -1
log "BOOTSTRAP DONE"
