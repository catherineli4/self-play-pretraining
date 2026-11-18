| size | arm | setting | best lr | best wd | zero-shot bpb | converged bpb | tokens to converge (M) | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1m | scratch | emb0.1 | 0.01 | 0.1 | 8.033 | 1.7356 | 1441 | 6 |
| 1m | scratch | emb0.3 | 0.01 | 0.1 | 8.033 | 1.7105 | 1468 | 6 |
| 1m | scratch | emb1.0 | 0.01 | 0.1 | 8.033 | 1.7725 | 1572 | 6 |
| 1m | scratch | head0 | 0.001 | 0 | 8.000 | 1.7388 | 1337 | 6 |
| 1m | scratch | resid0 | 0.003 | 0.1 | 8.032 | 1.7459 | 1572 | 6 |
| 1m | scratch | resid1 | 0.01 | 0.1 | 8.039 | 1.7228 | 1468 | 6 |
| 1m | scratch | std0.005 | 0.003 | 0.1 | 8.034 | 1.7096 | 1337 | 6 |
| 1m | scratch | std0.01 | 0.003 | 0.1 | 8.042 | 1.7263 | 1415 | 6 |
| 1m | scratch | std0.02 | 0.01 | 0.1 | 8.069 | 1.7388 | 1022 | 6 |
| 1m | scratch | std0.03 | 0.003 | 0.1 | 8.071 | 1.7558 | 1520 | 6 |
| 1m | scratch | std0.05 | 0.01 | 0.1 | 8.123 | 1.7468 | 1232 | 6 |
| 1m | scratch | std0.1 | 0.01 | 0.1 | 8.122 | 1.7397 | 1179 | 6 |
| 1m | warm | - | 0.01 | 0 | 9.059 | 1.6851 | 760 | 6 |
| 1m | warm | r2800 | 0.003 | 0.1 | 6.646 | 1.6947 | 1022 | 6 |
wrote /mnt/localssd/spp-pp1m/results.csv
