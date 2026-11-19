# Results (1m learner, one seed)

Small summaries only; full logs, checkpoints and per-run histories are in
`gs://cmu-gpucloud-catheri4/self-play-pretraining/` and in the Weights & Biases projects
`self-play-pretraining`, `self-play-prepretraining-1m-dclm`, `self-play-prepretraining-1m-fresh`.

- `1m_replication/`: zero-shot bits-per-byte curves of the self-play run (`sp_1m_seed0`, 8192 rounds, 43B synthetic
  tokens) and the uniform-prior baseline on DCLM and Metamath, evaluated on every 100-round checkpoint
  (`*_first256` = first 256 records of each file, as the trainer's own eval; `*_random256` = 256 random records).
- `1m_prepretraining_plateau/`: paper App. A protocol (train until the validation bpb plateaus) on the 270 MB DCLM
  stream, from scratch over Gaussian initialisation scales x LR x WD, and warm-started from self-play checkpoints.
- `1m_prepretraining_fresh/`: the same arms trained for exactly one pass over the first 1B and 2B bytes of a 2B-byte
  DCLM stream (no repetition), validated on a disjoint held-out stream.

The 25m pipeline (`scripts/run_25m_pipeline.sh`) writes its results to `gs://.../pp25m/` when it finishes.

- `1m_reward_variant/`: ablation of the learning-progress reward without the AdamW step operator at 1m (4096 rounds);
  `25m/reward_variant.md` is the same ablation at 25m for the full schedule.
- `1m_generator_variant/`: ablation where the generator is updated on its own fresh samples only (no policy-gradient or
  expert-iteration fine-tuning on replayed or mutated programs from earlier rounds), 1m, 4096 rounds.
- `plots/`: every figure as SVG, grouped by experiment (`plots/README.md` lists them with captions; `plots/index.html`
  shows them all on one page). Regenerate with `python scripts/plot_results.py`; the training-dynamics and
  validation-curve figures need the local run logs, the rest come from the files in this directory.
- `25m/`: the 25m self-play run's zero-shot curve (every 50 rounds) and the fresh-DCLM downstream comparison of
  warm starts (self-play checkpoints at 1.05B, 2.1B and 34.4B synthetic tokens; the k-Dyck, NCA and DeclRef
  curricula of "Rethinking Pre-pretraining" at 2B synthetic tokens) against training from scratch, one pass over
  1B and 2B fresh DCLM bytes. Compute-matched from-scratch baselines (3.05B, 4.1B, 36.4B fresh tokens) are added
  by `scripts/run_matched_compute.sh`.
