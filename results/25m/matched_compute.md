# Compute-matched baselines (25m learner, fresh DCLM, one epoch)

Each scratch run trains the default init (std 0.02) on as many fresh DCLM tokens as the warm-started
arm consumed in total: its synthetic tokens plus the 2B downstream budget. Same single-epoch protocol
(constant LR, 200-step cosine decay at the end), WD 0.1, best of LR {1e-3, 3e-3} for the two small
budgets; the 36.4B run used LR 1e-3 only (4-GPU DDP, 4.9 h, 138,888 steps; curve in
`scratch_36.4B_fresh_dclm_history.jsonl`). The three curricula consumed 2B synthetic tokens, so their
matched budget is 4B; the 4.1B run stands in for it. Full per-run table: `downstream_fresh_dclm_summary.md`.

| Warm start | Synthetic tokens | Warm start + 2B fresh, bpb | Matched scratch budget | Scratch, bpb | Scratch minus warm |
| --- | --- | --- | --- | --- | --- |
| Self-play round 200 | 1.05B | 1.368 | 3.05B | 1.374 | +0.006 |
| Self-play round 400 | 2.1B | 1.374 | 4.1B | 1.357 | -0.017 |
| Self-play final (round 6560) | 34.4B | 1.380 | 36.4B | 1.289 | -0.091 |
| DeclRef | 2B | 1.360 | 4.1B | 1.357 | -0.003 |
| NCA | 2B | 1.399 | 4.1B | 1.357 | -0.042 |
| k-Dyck | 2B | 1.415 | 4.1B | 1.357 | -0.058 |

Only the earliest self-play checkpoint survives matched compute, by 0.006 bpb, less than the 0.02 spread
between the two learning rates. Every later self-play arm and every curriculum is beaten by spending the
same compute on fresh DCLM; at the full budget the gap is 0.091 bpb. DeclRef is a tie. The 36.4B run's
constant-LR curve passed 1.407 at 3.05B and 1.393 at 4.1B tokens, so the final decay is worth about
0.035 bpb at these budgets, and the separate decayed runs are the right comparison. One seed throughout.
