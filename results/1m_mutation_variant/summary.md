# Ablation: no mutations (1m learner, one seed)

Arrows: ↓ lower is better, ↑ higher is better; training-dynamics quantities are descriptive and carry neither.

`sp_1m_nomut_seed0`: the replication config and seed (`configs/replicate_1m.json`), 4096 rounds, 8 GPUs, `--mut_frac 0`.
In the paper's setting the round pool is 896 fresh samples + 128 mutations of archive programs + 256 programs replayed
from the bank. The variant drops the mutations: all 1024 non-replay programs are fresh generator samples, replay stays at
256, and both generator losses (policy gradient and expert iteration) run over the whole pool as in the baseline. The
archive is still filled but never sampled, and the replay bank receives only generator samples.
Baseline = `sp_1m_seed0` at matching rounds; the two earlier 1m ablations, `sp_1m_rawdelta_seed0` (no AdamW step
operator, `../1m_reward_variant`) and `sp_1m_genfresh_seed0` (generator on fresh samples only, `../1m_generator_variant`),
are shown alongside. Files: `sp_1m_nomut_seed0_config.json`, `sp_1m_nomut_seed0_eval_curve_heldout256.jsonl`
(last 256 records, `scripts/eval_checkpoints.py`), `sp_1m_nomut_seed0_eval_curve_first256.jsonl` (trainer evals + ICL).
Wall clock 59 min at 0.82 s/round. Plots: `../plots/1m_mutation_variant/`.

## Warm-start value (fresh DCLM, one epoch, best of LR {3e-3, 1e-2} x WD {0, 0.1}; final bpb after the 200-step decay)

| Warm start | Self-play rounds | 1B fresh tokens ↓ | 2B fresh tokens ↓ | Tokens to 1.75 bpb (2B run) ↓ |
| --- | --- | --- | --- | --- |
| Scratch, std 0.02 | 0 | 1.804 | 1.751 | not reached |
| Baseline round 2800 | 2800 | 1.714 | 1.687 | 943M |
| Baseline round 4100 | 4100 | 1.712 | 1.686 | 970M |
| No step operator, round 4096 | 4096 | 1.707 | 1.678 | 839M |
| Generator on fresh only, round 4096 | 4096 | 1.730 | 1.704 | 1337M |
| No mutations, round 2800 | 2800 | 1.736 | 1.704 | 1337M |
| No mutations, round 4096 | 4096 | 1.725 | 1.698 | 1310M |

Full per-run table: `../1m_prepretraining_fresh/summary.md` (arms `warm_nomut_r2800`, `warm_nomut_r4096`).

## Zero-shot transfer and in-context learning (rounds up to 4100)

| Metric | Baseline | No step operator | Generator on fresh only | No mutations |
| --- | --- | --- | --- | --- |
| DCLM bpb (held out), mean over rounds 3100-4100 ↓ | 7.48 | 7.70 | 7.25 | 6.56 |
| DCLM bpb (held out), best checkpoint ↓ | 6.63 | 6.82 | 6.43 | 5.77 |
| Metamath bpb (held out), mean over rounds 3100-4100 ↓ | 6.33 | 6.57 | 6.55 | 5.99 |
| Metamath bpb (held out), best checkpoint ↓ | 5.92 | 6.01 | 6.06 | 5.51 |
| Arithmetic bpb (trainer eval), mean over rounds 3100-4100 ↓ | 0.51 | 1.53 | 1.70 | 0.81 |
| Arithmetic bpb, best checkpoint ↓ | 0.25 | 0.79 | 0.79 | 0.53 |
| ICL stack, 32 shots, round 4000 (best) ↑ | 0.85 (0.92) | 0.70 (0.78) | 0.77 (0.77) | 0.71 (0.71) |
| ICL associative recall, 32 shots, round 4000 (best) ↑ | 0.77 (0.84) | 0.34 (0.51) | 0.30 (0.44) | 0.65 (0.79) |
| ICL reverse, 32 shots, round 4000 (best) ↑ | 0.34 (0.92) | 0.00 (0.01) | 0.48 (0.96) | 0.02 (0.02) |
| ICL max / min, 32 shots, round 4000 ↑ | 0.24 / 0.22 | 0.12 / 0.13 | 0.20 / 0.20 | 0.20 / 0.16 |

## Training dynamics (means over rounds 3800-4095)

| Quantity | Baseline | No step operator | Generator on fresh only | No mutations |
| --- | --- | --- | --- | --- |
| Learner loss on the pool (nats/byte) | 0.38 | 0.17 | 0.74 | 2.02 |
| Learner loss, fresh programs / replayed programs | 0.28 / 0.46 | 0.04 / 0.19 | 0.70 / 0.83 | 2.02 / 2.00 |
| Bytes emitted per program | 1036 | 485 | 1411 | 3239 |
| Programs filling the whole output | 23% | 10% | 33% | 77% |
| Programs with empty output | 24% | 29% | 59% | 10% |
| Program length (bytes) | 31 | 33 | 14 | 35 |
| Loop nesting depth | 2.05 | 1.70 | 1.02 | 2.60 |
| Generator nats per program token | 2.56 | 2.56 | 1.49 | 2.11 |
| Generator KL to the uniform prior, whole pool (nats/program) | 15.6 | 14.7 | 9.7 | 36.0 |
| Mean importance ratio on PG rows | 6e4 | 4e5 | 1.0 | 0.99 |
| Generator PG loss magnitude | 7e6 | 4e7 | 25 | 129 |
| Generator gradient norm before clipping | 6e6 | 2e7 | 5.7 | 4.7 |
| Reward mean, fresh programs | 2.1 | 0.004 | 5.4 | 10.8 |
| Occupied archive niches | 33 | 32 | 33 | 27 |

The no-step-operator run's reward is a raw directional derivative on a different scale, so its reward and PG-loss rows
are not comparable to the others.

## Where the e^20 importance ratios come from

In the baseline, the policy-gradient loss is around 1e7 because replayed rows carry importance ratios clipped at e^20.
This run shows that those ratios come from mutations. A mutated program enters the replay bank with the generator's
log-probability at that round, which is very low because the generator never sampled it. Expert iteration then pulls the
generator toward the rewarding mutants, so when one is replayed later its ratio is enormous. With no mutations the bank
holds only on-policy samples, the mean ratio stays at 0.99, and the PG loss and gradient norm are ordinary (about 130
and 5). After gradient clipping the baseline's generator update is therefore steered mostly by a few replayed mutants;
here it is steered by the whole pool.

## Reading

Removing mutations costs some warm-start value, less than the generator ablation: 0.013 bpb at 1B and 0.012 at 2B
against the baseline at the same round (1.725 vs 1.712, 1.698 vs 1.686), with 1.75 bpb reached 35% later. It stays
0.08 and 0.05 bpb ahead of scratch. At round 2800 the gap is wider (0.022 and 0.017), so the variant was still catching
up when it stopped.

The curriculum changes character. Without mutants to imitate, the generator moves much further from the prior (KL 36
vs 16 nats per program) toward long, deeply nested programs whose output fills the whole output budget (77% vs 23%, 3.1x
the bytes). The learner finds these streams hard to predict (pool loss 2.0 vs 0.4 nats/byte), and the reward on fresh
programs is 5x the baseline's. This is the best run so far on zero-shot natural text: held-out DCLM 6.56 vs 7.48 bpb
and Metamath 5.99 vs 6.33. That gain does not carry into warm-start value, which is the same pattern as the generator
ablation: zero-shot bpb on natural text is a poor predictor of fine-tuning value at 1m.

In-context learning keeps associative recall (0.65 vs 0.77) but loses reverse entirely (0.02 at every checkpoint vs a
best of 0.92), and arithmetic is worse (0.81 vs 0.51 bpb). The mutation operator seems to be what injects the short,
structured programs behind those skills; the generator on its own drifts toward high-volume, hard-to-predict output.

Across the three 1m ablations, the step operator is not needed, while both the generator's fine-tuning on earlier
programs and the mutations contribute warm-start value. One seed, 4096 of 8192 rounds; the 0.012 to 0.013 gap is larger
than the LR-grid spread between baseline checkpoints (0.002 to 0.005) but seeds have not been run.
