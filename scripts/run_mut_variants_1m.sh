#!/usr/bin/env bash
# Variants of the 1m mutation ablation, run one after another (each: 4096 rounds of self-play on 8 GPUs, then the
# fresh-DCLM warm-start sweep of rounds 2800 and 4096), via scripts/run_mut_variant_1m.sh. Re-entrant: finished
# variants are skipped, a preempted one resumes.
#   mutnobank  : mutations as usual, but never banked (--mut_to_bank 0): replay is on-policy, no e^20 importance ratios
#   mutlearner : mutations train the learner only (--mut_to_bank 0 --gen_update no_mut): not banked, not in the EI loss
#   mut25      : twice the mutations (--mut_frac 0.25: 768 fresh + 256 mutants)
#   mut06      : half the mutations (--mut_frac 0.0625: 960 fresh + 64 mutants)
# Launch: setsid nohup bash scripts/run_mut_variants_1m.sh > /mnt/localssd/spp-runs/run_mut_variants_1m.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
VARIANTS=${VARIANTS:-"mutnobank mutlearner mut25 mut06"}
declare -A FLAGS=([mutnobank]="--mut_to_bank 0" [mutlearner]="--mut_to_bank 0 --gen_update no_mut"
                  [mut25]="--mut_frac 0.25" [mut06]="--mut_frac 0.0625")
PP=${PP:-/mnt/localssd/spp-pp1m-fresh}
for v in $VARIANTS; do
  if grep -q "^| 2B | warm_${v}_r4096 " "$PP/summary.md" 2>/dev/null; then
    echo "[$(date -u '+%F %T')] $v already done"; continue
  fi
  echo "[$(date -u '+%F %T')] variant $v: ${FLAGS[$v]}"
  TAG=$v EXTRA="${FLAGS[$v]}" bash scripts/run_mut_variant_1m.sh || { echo "[$(date -u '+%F %T')] variant $v failed"; exit 1; }
done
echo "[$(date -u '+%F %T')] ALL DONE"
