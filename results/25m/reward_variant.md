# Ablation: learning-progress reward without the AdamW step operator (25m learner, full schedule, one seed)

Arrows: ↓ lower is better, ↑ higher is better; training-dynamics quantities are descriptive and carry neither.

`sp_25m_rawdelta_seed0`: the 25m replication config and seed (`configs/replicate_25m.json`, 6560 rounds, 34.4B
synthetic tokens), 8 GPUs, `--tangent_precond identity`. The reward tangent is theta_past - theta_e instead of
P_e (theta_past - theta_e) with P_e = lr / (sqrt(v_hat_e) + eps) (paper Eq. 2); rewards are z-scored across the
pool, so only the per-coordinate weighting changes. Baseline = `sp_25m_seed0` (same config, 7 GPUs). Config:
`sp_25m_rawdelta_seed0_config.json`; trainer eval curve: `sp_25m_rawdelta_seed0_eval_curve_first256.jsonl`.
Wall clock 8.3 h at 4.45 s/round (baseline 9.2 h at 4.94 s/round on 7 GPUs). The 1m version of this ablation
(4096 of 8192 rounds) is in `../1m_reward_variant/summary.md`.

## Warm-start value (fresh DCLM, one epoch, best of LR {1e-3, 3e-3}, WD 0.1; final bpb after the 200-step decay)

| Warm start | Self-play rounds | Synthetic tokens | 1B fresh tokens ↓ | 2B fresh tokens ↓ | Tokens to 1.45 bpb (2B run) ↓ |
| --- | --- | --- | --- | --- | --- |
| Scratch, std 0.02 | 0 | 0 | 1.444 | 1.388 | 1389M |
| Baseline round 200 | 200 | 1.05B | 1.417 | 1.368 | 1075M |
| Baseline round 400 | 400 | 2.1B | 1.424 | 1.374 | 1153M |
| Baseline final | 6560 | 34.4B | 1.420 | 1.380 | 1258M |
| Variant round 200 | 200 | 1.05B | 1.417 | 1.366 | 1048M |
| Variant round 400 | 400 | 2.1B | 1.429 | 1.374 | 1153M |
| Variant final | 6560 | 34.4B | 1.412 | 1.372 | 1101M |

Full per-run table: `downstream_fresh_dclm_summary.md` (arms `sp_rawdelta_1B`, `sp_rawdelta_2B`, `sp_rawdelta_final`).
Against the compute-matched scratch runs of `matched_compute.md`: variant round 200 + 2B (3.05B total) 1.366 vs
scratch 1.374 (+0.008); round 400 1.374 vs 1.357 (-0.017); final 1.372 vs 1.289 (-0.083). As for the baseline,
only the earliest checkpoint survives matched compute.

## Zero-shot transfer and in-context learning

Trainer evals every 50 rounds (first 256 records of each file, the same protocol in both runs); ICL accuracy with
32 in-context examples, averaged over the five ICL evals at rounds 5000, 5500, 6000, 6500 and 6559.

| Metric | Baseline | Variant |
| --- | --- | --- |
| DCLM bpb, mean over rounds 5560-6559 ↓ | 5.99 | 6.01 |
| DCLM bpb, best checkpoint ↓ | 5.55 (round 6450) | 5.53 (round 6559) |
| DCLM bpb, last checkpoint ↓ | 5.86 | 5.53 |
| Metamath bpb, mean over rounds 5560-6559 ↓ | 3.76 | 3.71 |
| Metamath bpb, best checkpoint ↓ | 3.57 | 3.57 |
| Metamath bpb, last checkpoint ↓ | 3.67 | 3.59 |
| Arithmetic bpb, mean over rounds 5560-6559 ↓ | 0.18 | 0.08 |
| ICL stack, 32 shots ↑ | 0.88 | 0.85 |
| ICL associative recall, 32 shots ↑ | 0.85 | 0.89 |
| ICL reverse, 32 shots ↑ | 0.83 | 0.96 |
| ICL reverse, 8 shots ↑ | 0.38 | 0.90 |
| ICL max / min, 32 shots ↑ | 0.25 / 0.26 | 0.25 / 0.26 |
| ICL sum, 32 shots ↑ | 0.00 | 0.00 |

Zero-shot bpb by 1000-round window (baseline / variant):

| Rounds | DCLM ↓ | Metamath ↓ | Arithmetic ↓ |
| --- | --- | --- | --- |
| 0-999 | 7.74 / 8.02 | 7.22 / 8.09 | 6.61 / 8.02 |
| 1000-1999 | 7.03 / 7.51 | 4.89 / 5.56 | 0.33 / 1.83 |
| 2000-2999 | 7.17 / 7.69 | 4.50 / 4.83 | 0.30 / 0.20 |
| 3000-3999 | 7.08 / 7.08 | 4.64 / 4.31 | 0.30 / 0.14 |
| 4000-4999 | 6.38 / 6.82 | 4.05 / 4.22 | 0.22 / 0.10 |
| 5000-5999 | 6.18 / 6.26 | 3.83 / 3.79 | 0.18 / 0.07 |
| 6000-6559 | 5.92 / 5.95 | 3.74 / 3.69 | 0.18 / 0.08 |

## Training dynamics

Means over rounds 6060-6559 (last 500), and the learner's pool loss and emitted bytes by 1000-round window.

| Quantity | Baseline | Variant |
| --- | --- | --- |
| Learner loss on the pool (nats/byte) | 0.25 | 0.34 |
| Bytes emitted per program | 846 | 939 |
| Programs filling the whole output | 17% | 19% |
| Programs with empty output | 14% | 27% |
| Program depth | 2.18 | 2.28 |
| Generator KL to the uniform prior (nats/program) | 14.0 | 16.4 |
| Reward mean (raw directional derivative) | 17.3 | 0.038 |

| Rounds | Pool loss, baseline / variant | Emitted bytes, baseline / variant |
| --- | --- | --- |
| 0-999 | 0.11 / 0.10 | 339 / 372 |
| 1000-1999 | 0.11 / 0.12 | 361 / 383 |
| 2000-2999 | 0.14 / 0.15 | 546 / 494 |
| 3000-3999 | 0.15 / 0.36 | 615 / 920 |
| 4000-4999 | 0.17 / 0.34 | 645 / 914 |
| 5000-5999 | 0.20 / 0.34 | 691 / 923 |
| 6000-6559 | 0.25 / 0.34 | 835 / 923 |

## Reading

At 25m the step operator makes little difference, and what differs goes the other way from the 1m run. The
variant's weights are equal or slightly better warm starts (final checkpoint 0.008 bpb better at both budgets, round
200 within 0.002, round 400 equal at 2B and 0.005 worse at 1B); every self-play arm still beats scratch by 0.014 to
0.028 bpb at 1B and 0.014 to 0.022 at 2B. Zero-shot transfer ends level (DCLM 5.99 vs 6.01, Metamath 3.76 vs 3.71
over the last 1000 rounds) after lagging by 0.3 to 0.7 bpb for the first 3000 rounds, in-context learning is equal
or better (reverse 0.96 vs 0.83 at 32 shots, 0.90 vs 0.38 at 8), and the curriculum is harder rather than easier:
the variant's generator jumps to a pool loss of 0.35 around round 3000 and stays there, with 10% more emitted bytes,
twice the empty programs and a more specialised generator (KL to prior 16.4 vs 14.0), while the baseline's pool loss
rises gradually from 0.11 to 0.25. The 1m deficits (zero-shot 0.2 bpb worse, ICL much weaker at rounds 3100-4100 of
a 4096-round run) look like the same early lag, measured before the variant caught up; the 1m mechanism guess that
P_e lengthens outputs and improves ICL by up-weighting rarely used coordinates is not supported here. So the AdamW
step operator is not needed for the learning-progress reward to work at 25m; it mainly changes how fast the
generator's curriculum hardens. One seed each; the downstream gaps (0.002 to 0.008) are within the 0.01 spread
between the two learning rates, and the seed spread has not been measured.
