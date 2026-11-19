# Ablation: generator updated on its own fresh samples only (1m learner, one seed)

`sp_1m_genfresh_seed0`: the replication config and seed (`configs/replicate_1m.json`), 4096 rounds, 8 GPUs, `--gen_update fresh`.
In the paper's setting the round pool is 896 fresh samples + 128 mutations of archive programs + 256 programs replayed
from the bank, and the generator is updated on all of it: policy gradient on fresh + replayed rows (the replayed rows carry
importance ratios clipped at e^20) and expert iteration (reward-weighted likelihood) on every row. The variant restricts
both losses, including the advantage z-scoring and the EI weights, to the 896 fresh samples, so the generator never
fine-tunes on a program from an earlier round; mutations and replays still enter the learner's batch unchanged.
Baseline = `sp_1m_seed0` at matching rounds; `sp_1m_rawdelta_seed0` (no AdamW step operator, `../1m_reward_variant`)
is shown alongside. Files: `sp_1m_genfresh_seed0_config.json`, `sp_1m_genfresh_seed0_eval_curve_heldout256.jsonl`
(last 256 records, `scripts/eval_checkpoints.py`), `sp_1m_genfresh_seed0_eval_curve_first256.jsonl` (trainer evals + ICL).
Wall clock 52 min at 0.74 s/round. Plots: `../plots/1m_generator_variant/`.

## Warm-start value (fresh DCLM, one epoch, best of LR {3e-3, 1e-2} x WD {0, 0.1}; final bpb after the 200-step decay)

| Warm start | Self-play rounds | 1B fresh tokens | 2B fresh tokens | Tokens to 1.75 bpb (2B run) |
| --- | --- | --- | --- | --- |
| Scratch, std 0.02 | 0 | 1.804 | 1.751 | not reached |
| Baseline round 2800 | 2800 | 1.714 | 1.687 | 943M |
| Baseline round 4100 | 4100 | 1.712 | 1.686 | 970M |
| Baseline final | 8192 | 1.714 | 1.685 | 970M |
| No step operator, round 4096 | 4096 | 1.707 | 1.678 | 839M |
| Generator on fresh only, round 2800 | 2800 | 1.730 | 1.702 | 1284M |
| Generator on fresh only, round 4096 | 4096 | 1.730 | 1.704 | 1337M |

Full per-run table: `../1m_prepretraining_fresh/summary.md` (arms `warm_genfresh_r2800`, `warm_genfresh_r4096`).

## Zero-shot transfer and in-context learning (rounds up to 4100)

| Metric | Baseline | No step operator | Generator on fresh only |
| --- | --- | --- | --- |
| DCLM bpb (held out), mean over rounds 3100-4100 | 7.48 | 7.70 | 7.25 |
| DCLM bpb (held out), best checkpoint | 6.63 | 6.82 | 6.43 |
| Metamath bpb (held out), mean over rounds 3100-4100 | 6.33 | 6.57 | 6.55 |
| Metamath bpb (held out), best checkpoint | 5.92 | 6.01 | 6.06 |
| Arithmetic bpb (trainer eval), mean over rounds 3100-4100 | 0.51 | 1.53 | 1.70 |
| Arithmetic bpb, best checkpoint | 0.25 | 0.79 | 0.79 |
| ICL stack, 32 shots, round 4000 (best) | 0.85 (0.92) | 0.70 (0.78) | 0.77 (0.77) |
| ICL associative recall, 32 shots, round 4000 (best) | 0.77 (0.84) | 0.34 (0.51) | 0.30 (0.44) |
| ICL reverse, 32 shots, round 4000 (best) | 0.34 (0.92) | 0.00 (0.01) | 0.48 (0.96) |
| ICL max / min, 32 shots, round 4000 | 0.24 / 0.22 | 0.12 / 0.13 | 0.20 / 0.20 |

## Training dynamics (means over rounds 3800-4095)

| Quantity | Baseline | No step operator | Generator on fresh only |
| --- | --- | --- | --- |
| Learner loss on the pool (nats/byte) | 0.38 | 0.17 | 0.74 |
| Learner loss, fresh programs / replayed programs | 0.28 / 0.46 | 0.04 / 0.19 | 0.70 / 0.83 |
| Bytes emitted per program | 1036 | 485 | 1411 |
| Programs filling the whole output | 23% | 10% | 33% |
| Programs with empty output | 24% | 29% | 59% |
| Program length (bytes) | 31 | 33 | 14 |
| Loop nesting depth | 2.05 | 1.70 | 1.02 |
| Generator nats per program token | 2.56 | 2.56 | 1.49 |
| Generator KL to the uniform prior, whole pool (nats/program) | 15.6 | 14.7 | 9.7 |
| Generator PG loss magnitude | 7e6 | 4e7 | 25 |
| Generator gradient norm before clipping | 6e6 | 2e7 | 5.7 |

The KL statistic is computed over the whole pool, including the replayed and mutated rows that the variant's generator
never fits, so it understates how close the variant's own samples sit to the prior. The baseline's PG loss and gradient
norm are enormous because replayed rows carry importance ratios up to e^20: after gradient clipping, those rows set the
direction of the generator update in the paper's setting.

## Reading

This is the first ablation that costs warm-start value: 0.018 bpb at both budgets against the baseline at the same round
(1.730 vs 1.712 at 1B, 1.704 vs 1.686 at 2B), with 1.75 bpb reached 38% later, though still 0.05 to 0.07 bpb ahead of
scratch. Without fine-tuning on earlier programs the generator stays near the prior (1.5 nats per token vs 2.6, programs
half as long, no nested loops), its own samples are mostly trivial (59% empty outputs) or run to the output limit (33%),
and the learner's batch is hard on paper (pool loss 0.74 vs 0.38) because the replayed and mutated programs it still
trains on are never consolidated by the generator. Zero-shot DCLM is slightly better and Metamath and arithmetic worse;
in-context learning loses associative recall (0.30 vs 0.77) and keeps reverse and stack. So the expert-iteration and
replay terms, which pull the generator toward programs that were rewarding in earlier rounds, are what make the
curriculum compound; the reward's step operator (previous ablation) is not. One seed, 4096 of 8192 rounds; the 0.018
gap is larger than the LR-grid spread (0.002 to 0.005 between the baseline checkpoints) but seeds have not been run.
A stricter variant, dropping expert iteration entirely (`--lambda_ei 0` on top of `--gen_update fresh`), was not run.
