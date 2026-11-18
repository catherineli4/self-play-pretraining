# Self-Play Pretraining with Zero Data

Reference implementation of [arXiv:2609.30063](https://arxiv.org/abs/2609.30063)
(Cowsik, Dolev, Li, De Luca, Cohen, Goodman, Levine). A **generator** LM writes
programs for a Brainf*ck-like universal machine, the machine executes them into byte
sequences, and a **learner** LM is pretrained on those bytes by next-byte prediction.
The generator is trained by policy gradient with a *learning-progress* reward: the
absolute AdamW-preconditioned alignment between a program's learner gradient and the
learner's parameter movement since round floor(e/2), computed with forward-mode AD.
No natural data is used for any gradient update.

The paper ships no code; this is written from Section 2 and Appendices A-H of the
TeX source. See `DEVIATIONS` below for the choices the paper leaves open.

## Layout

| Path | Contents |
| --- | --- |
| `spp/tokens.py` | byte vocabulary, 19-token program alphabet, macro table |
| `spp/bf.c` / `spp/bf.py` | universal machine (C + OpenMP, Python reference), batch execution |
| `spp/model.py` | Llama-style decoder, presets `tiny 1m 3m 8m 25m` |
| `spp/reward.py` | Eq. 2 reward via `torch.func.jvp`, AdamW preconditioner |
| `spp/generator.py` | restricted sampling, sequence log-probs, Eqs. 3-5 losses |
| `spp/bank.py` | replay bank, MAP-Elites archive, mutations (App. G) |
| `spp/train.py` | the self-play loop, uniform-prior and PCFG baselines, checkpoints, W&B, GCS |
| `spp/data.py` | byte-record evaluation (bits/byte), synthetic evals, uniform prior, random PCFG |
| `spp/icl.py` | six ICL tasks + greedy-success evaluation |
| `spp/structure.py` | detector for arithmetic/quadratic/cubic/Fibonacci/geometric outputs (App. D) |
| `scripts/` | `train.py`, `evaluate.py`, `dashboard.py`, `analyze_programs.py`, `prepare_data.py`, `launch_replication.sh` |
| `tests/test_all.py` | unit tests (`python -m unittest discover -s tests`) |

## Quick start

```bash
PY=/home/catherine/miniconda3/envs/optim-study/bin/python   # any env with torch>=2.3 and numpy
$PY -m unittest discover -s tests                            # ~3 s on a GPU

# eval data (Metamath set.mm at the paper's pinned commit -> metamath.bin)
$PY scripts/prepare_data.py metamath /mnt/localssd/spp-data

# the 1M-parameter replication on GPUs 0-6 (torchrun), the uniform-prior baseline on GPU 7, W&B + GCS + dashboard
wandb login                     # once (or export WANDB_API_KEY=..., or WANDB_MODE=offline)
bash scripts/launch_replication.sh
#   GPUS_SP=0,1,2,3,4,5,6 GPU_UNIFORM=7 RUNS=... DATA=... WANDB_PROJECT=... GCS_PATH=gs://bucket/prefix PORT=8787 SEED=0
#   all eight GPUs on self-play, no baseline:  GPUS_SP=0,1,2,3,4,5,6,7 GPU_UNIFORM= bash scripts/launch_replication.sh

# multi-GPU by hand: the round pool is sharded across processes for the reward JVP and the learner step (DDP)
torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/replicate_1m.json --out_dir runs/sp1m

# single run by hand
$PY scripts/train.py --config configs/replicate_1m.json --out_dir runs/sp1m \
    --eval metamath=/mnt/localssd/spp-data/metamath.bin --wandb_project self-play-pretraining \
    --gcs_path gs://bucket/prefix
$PY scripts/dashboard.py --runs runs --port 8787      # http://localhost:8787 (ssh -L 8787:localhost:8787 host)

# evaluate a checkpoint: zero-shot bits/byte at 256 context + ICL curves
$PY scripts/evaluate.py runs/sp1m/ckpt_last.pt --eval metamath=/mnt/localssd/spp-data/metamath.bin --icl
# earliest round per discovered math family (from runs/sp1m/programs/)
$PY scripts/analyze_programs.py runs/sp1m
```

Every `TrainConfig` field is a CLI flag (`--preset 3m --rounds 2000 --beta_kl 0.1 ...`);
`--data uniform` / `--data pcfg` train the learner on the fixed baselines instead.
`--reward_variant {abs,signed,negate,shuffle,loss_delta}` and `--lookback last_step`
reproduce the reward ablations of App. F.

W&B: `--wandb_project` logs every round (losses, rewards by source, program and archive
stats, timings, zero-shot BPB at the training context and at 256 bytes, ICL success and
probability curves, reward/length/depth histograms, a table of the top programs and the
detected math families every `sample_every` rounds). The launch script falls back to
`WANDB_MODE=offline` if no API key is configured; `scripts/wandb_sync.sh` then pushes the
offline runs to the cloud every 5 minutes once `wandb login` has been run.

Checkpoints: `ckpt_last.pt` is resumable (`--resume`) and holds both models, both
optimizers, the bank and the archive; `models_round_XXXXXX.pt` holds just the learner
and generator weights. With `--gcs_path` both are mirrored to `gs://.../<run name>/`.

## Cost and speed

Per round, 1M preset, 4096 context, the paper's pool of 1024 fresh + 256 replay
programs (5.2M tokens), H100s:

| setup | reward JVP | learner step | execution | round |
| --- | --- | --- | --- | --- |
| 1 GPU | ~16 s | ~1.1 s | 0.2 s | ~18 s |
| 7 GPUs (torchrun, pool sharded) | 2.7 s | 0.3 s | 0.2 s | 3.25 s |

The reward dominates because forward-mode AD can only use PyTorch's unfused math
attention backend. `configs/replicate_1m.json` (8192 rounds, 43B tokens) takes about
7.5 h on 7 GPUs. Two further levers:

- **JVP flash-attention kernel** (what the paper used): `pip install jvp_flash_attention`
  (torch>=2.8, Triton; or put it in a directory and set `SPP_PYLIBS=/that/dir`).
  `spp/fastattn.py` picks it up automatically (`--jvp_kernel auto|flash|math`).
  Measured: 10.6x faster reward than the math backend, directional-derivative
  correlation 0.9999999. With it, a 7-GPU round is about 1 s. Triton compiles into
  `TRITON_CACHE_DIR` (the launch script puts it on the SSD; the home disk here is small).
- **Sharding, replay and the generator stay exact**: rank 0 owns the generator, bank,
  archive, lookback checkpoints, evaluation and checkpointing; the other ranks only
  compute their shard of the reward and the learner gradient. `tests/ddp_check.py`
  (run with torchrun) verifies the sharded reward is identical to the single-process
  one and the sharded AdamW step matches to 1e-6.

## DEVIATIONS and open choices

The paper does not state these; the defaults here are guesses, all configurable:
step budget (65536), tape size (1024 cells), max program length (64 tokens),
mutation fraction (1/8 of the fresh slots), replay size (128 or 256), KL coefficient
beta (0.05), learning rates (3e-3, generator ratio 1.0), warmup (100 rounds), and the
exact layer counts per preset. The reward uses floor(e/2) as in Eq. 2 (the text also
mentions ceil(e/2)). Mutated programs enter the replay bank with their evaluated
log-prob under the generator at creation time, so they can later be replayed with an
importance ratio. Evaluations on DCLM, DNA, speech, audio and Mutopia need their
datasets prepared as byte files (`--eval name=path`); only Metamath, CIFAR-10 and
plain text have download helpers. Epiplexity (Sec. 3.1) is not implemented.

## Pre-pretraining experiments (paper Appendix A) and spot-instance recovery

- `scripts/pretrain_natural.py`: pretrain on a natural byte corpus from scratch (Gaussian init with
  `--init_std/--init_emb_std/--init_head_std/--init_resid_scale`, see `spp/init.py`) or warm-started
  from a self-play learner checkpoint (`--warm_start models_round_XXXXXX.pt`). Default protocol is the
  paper's: constant LR until validation bpb plateaus (< 0.005 improvement for 5 evaluations), then a
  200-step cosine decay; `--max_epochs 1 --corpus_tokens N --val_corpus FILE` instead trains exactly one
  pass over the first N bytes (fresh tokens, no repetition) with a separate held-out validation file.
- `scripts/sweep_pretrain.py` + `scripts/orchestrate_1m.sh`: the init-scale sweep (global std, embedding
  std, residual scale, LM head) x LR x WD, plus warm-start arms; `scripts/collect_pretrain.py` summarises.
- `scripts/sweep_fresh.py` + `scripts/run_fresh_epoch.sh`: the single-epoch comparison on 1B and 2B fresh
  DCLM bytes (`scripts/prepare_data.py dclm --mb_per_shard 200 --local_shards 0,1,2` builds the 2B stream,
  `--local_shards 9` a disjoint validation stream); `scripts/collect_fresh.py` summarises.
- `scripts/bootstrap_instance.sh`: recovers a fresh or preempted spot VM (local-SSD RAID, NVIDIA module via
  dkms for the running kernel, conda env with torch 2.8, data and checkpoints from GCS). Everything durable
  is mirrored to `gs://cmu-gpucloud-catheri4/self-play-pretraining/` (checkpoints every 100 rounds, logs,
  data, sweep results), so instances are disposable.
