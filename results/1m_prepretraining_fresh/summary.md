| budget | arm | best lr | best wd | zero-shot | bpb @10% | bpb @25% | bpb @50% | bpb @75% | bpb @100% | final (after decay) | tokens to 2.5 | tokens to 2.2 | tokens to 2.0 | tokens to 1.9 | tokens to 1.8 | tokens to 1.75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2B | scratch_std0.005 | 0.003 | 0.1 | 8.035 | 2.303 | 2.023 | 1.921 | 1.825 | 1.739 | **1.7390** | 131M | 262M | 603M | 1179M | 1730M | 1992M |
| 2B | scratch_std0.02 | 0.01 | 0.1 | 8.070 | 2.199 | 2.023 | 1.886 | 1.847 | 1.751 | **1.7506** | 105M | 183M | 629M | 891M | 1992M | - |
| 2B | warm_final | 0.01 | 0 | 9.034 | 1.908 | 1.797 | 1.746 | 1.726 | 1.685 | **1.6846** | 52M | 79M | 105M | 210M | 472M | 970M |
| 2B | warm_r2800 | 0.01 | 0 | 6.663 | 1.899 | 1.794 | 1.747 | 1.726 | 1.687 | **1.6866** | 52M | 52M | 105M | 183M | 446M | 943M |
| 1B | scratch_std0.005 | 0.003 | 0.1 | 8.035 | 2.796 | 2.183 | 2.033 | 1.978 | 1.850 | **1.8498** | 131M | 236M | 655M | 970M | - | - |
| 1B | scratch_std0.02 | 0.01 | 0.1 | 8.070 | 2.568 | 2.155 | 2.038 | 1.917 | 1.804 | **1.8040** | 105M | 210M | 603M | 865M | - | - |
| 1B | warm_final | 0.01 | 0 | 9.034 | 2.045 | 1.861 | 1.794 | 1.765 | 1.714 | **1.7137** | 52M | 52M | 105M | 183M | 498M | 970M |
| 1B | warm_r2800 | 0.01 | 0 | 6.663 | 2.049 | 1.862 | 1.793 | 1.764 | 1.714 | **1.7141** | 52M | 52M | 105M | 183M | 498M | 943M |

32 runs. Columns '@x%' = best validation bpb seen within the first x% of the epoch; 'tokens to' = fresh tokens consumed when validation bpb first reached the threshold.
wrote /mnt/localssd/spp-pp1m-fresh/results.csv
