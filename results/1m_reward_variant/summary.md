# Ablation: learning-progress reward without the AdamW step operator (1m learner, one seed)

`sp_1m_rawdelta_seed0`: the replication config and seed (`configs/replicate_1m.json`), 4096 rounds (half the
replication; the learner LR is constant after the 100-round warmup, so this is a prefix of the full schedule), 8 GPUs,
`--tangent_precond identity`. The reward tangent is theta_past - theta_e instead of P_e (theta_past - theta_e) with
P_e = lr / (sqrt(v_hat_e) + eps) (paper Eq. 2). Rewards are z-scored across the pool in the generator losses, so only the
per-coordinate weighting changes; raw reward magnitudes are about 100x smaller. Baseline = `sp_1m_seed0` at the matching
rounds (checkpoints every 100 rounds, so round 4100 stands in for 4096).

## Warm-start value (fresh DCLM, one epoch, best of LR {3e-3, 1e-2} x WD {0, 0.1}; final bpb after the 200-step decay)

| Warm start | Self-play rounds | 1B fresh tokens | 2B fresh tokens | Tokens to 1.75 bpb (2B run) |
| --- | --- | --- | --- | --- |
| Scratch, std 0.02 | 0 | 1.804 | 1.751 | not reached |
| Baseline round 2800 | 2800 | 1.714 | 1.687 | 943M |
| Baseline round 4100 | 4100 | 1.712 | 1.686 | 970M |
| Baseline final | 8192 | 1.714 | 1.685 | 970M |
| Variant round 2800 | 2800 | 1.710 | 1.683 | 917M |
| Variant round 4096 | 4096 | 1.707 | 1.678 | 839M |

Full per-run table: `../1m_prepretraining_fresh/summary.md` (arms `warm_rawdelta_r2800`, `warm_rawdelta_r4096`, `warm_r4100`).

## Zero-shot transfer and in-context learning (matched rounds)

DCLM and Metamath: `scripts/eval_checkpoints.py`, last 256 records of each file (held out from the trainer's own eval),
curves in `sp_1m_seed0_eval_curve_heldout256.jsonl` and `sp_1m_rawdelta_seed0_eval_curve_heldout256.jsonl`.
Arithmetic and ICL (accuracy with 32 in-context examples) from the trainer's log at round 4000.

| Metric | Baseline | Variant |
| --- | --- | --- |
| DCLM bpb, mean over rounds 3100-4100 | 7.48 | 7.70 |
| DCLM bpb, best checkpoint up to round 4100 | 6.63 | 6.82 |
| Metamath bpb, mean over rounds 3100-4100 | 6.33 | 6.57 |
| Metamath bpb, best checkpoint up to round 4100 | 5.92 | 6.01 |
| Arithmetic bpb at round 4000 (best up to 4100) | 0.82 (0.25) | 1.80 (0.79) |
| ICL stack, 32 shots, round 4000 | 0.85 | 0.70 |
| ICL associative recall, 32 shots | 0.77 | 0.34 |
| ICL reverse, 32 shots | 0.34 | 0.00 |

## Training dynamics (means over rounds 3800-4095)

| Quantity | Baseline | Variant |
| --- | --- | --- |
| Learner loss on the pool (nats/byte) | 0.38 | 0.17 |
| Bytes emitted per program | 1036 | 485 |
| Programs filling the whole output | 23% | 10% |
| Programs with empty output | 24% | 29% |
| Generator KL to the uniform prior (nats/program) | 15.6 | 14.7 |
| Reward mean (raw directional derivative) | 5.1 | 0.046 |

## Reading

Without the step operator the generator settles on an easier curriculum (half the output bytes, learner loss 0.17 vs 0.38),
zero-shot transfer is 0.1 to 0.2 bpb worse and in-context learning is much weaker, yet the weights are slightly better warm
starts (0.005 to 0.008 bpb at both budgets, and 1.75 bpb reached with 14% fewer fresh tokens). A plausible mechanism: AdamW
updates have roughly uniform per-coordinate magnitude, so theta_past - theta_e weights all coordinates alike, while
P_e up-weights coordinates with small gradient variance (such as embeddings of rare bytes) and so rewards programs that
exercise rarely used parts of the model, which shows up as longer, more varied outputs and better ICL. The warm-start
margin is single-seed and of the same order as the spread between baseline checkpoints (0.002) and the LR grid (0.01 to 0.03),
so it needs seeds before being called a gain; the zero-shot and ICL deficits are large relative to those spreads.

Update (25m, full 6560-round schedule, `../25m/reward_variant.md`): at 25m the variant ends level on zero-shot transfer and
equal or better on ICL after lagging for the first 3000 rounds, is a slightly better warm start (0.008 bpb at the final
checkpoint), and settles on a harder, not easier, curriculum. The 1m deficits above were measured at rounds 3100-4100 of a
4096-round run, inside that early lag, and the mechanism guess in the reading is not supported at 25m.
