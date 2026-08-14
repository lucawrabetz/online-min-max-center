#!/usr/bin/env bash
set -euo pipefail

HOST="$(hostname -s)"
LOGDIR="out/campaign_${HOST}"
mkdir -p "$LOGDIR" out logs

TABLE_GAMMAS=(3 5 10)
PANEL_B_GAMMAS=(1.2 1.3 6)

echo "=== Appendix D / $HOST / started $(date '+%F %T') ==="

echo "[$(date '+%F %T')] START  panelA_low"
python3 run_experiments.py \
  --set_name unitsquarefixedzero --solvers OMIP,SOMIP --perm none \
  --gamma_lo 0 --gamma_hi 2 --gamma_step 0.1 --force \
  >"$LOGDIR/panelA_low.log" 2>&1
echo "[$(date '+%F %T')] DONE   panelA_low"

echo "[$(date '+%F %T')] START  panelA_high"
python3 run_experiments.py \
  --set_name unitsquarefixedzero --solvers OMIP,SOMIP --perm none \
  --gamma_lo 2 --gamma_hi 12 --gamma_step 0.5 --force \
  >"$LOGDIR/panelA_high.log" 2>&1
echo "[$(date '+%F %T')] DONE   panelA_high"

for G in "${PANEL_B_GAMMAS[@]}"; do
  for T in 10 20 30 40 50; do
    echo "[$(date '+%F %T')] START  panelB_G${G}_T${T}"
    python3 run_experiments.py \
      --set_name unitsquarefixedzero --solvers OMIP,SOMIP --perm none \
      --gamma "$G" --T "$T" \
      >"$LOGDIR/panelB_G${G}_T${T}.log" 2>&1
    echo "[$(date '+%F %T')] DONE   panelB_G${G}_T${T}"
  done
done

for T in 50 75 100; do
  for G in "${TABLE_GAMMAS[@]}"; do
    echo "[$(date '+%F %T')] START  table_T${T}_G${G}_somip"
    python3 run_experiments.py \
      --set_name timestudyfixedzero --solvers SOMIP --perm none \
      --gamma "$G" --T "$T" \
      >"$LOGDIR/table_T${T}_G${G}_somip.log" 2>&1
    echo "[$(date '+%F %T')] DONE   table_T${T}_G${G}_somip"
  done
done

mv out/final.csv "out/${HOST}-appendixD-final.csv"
echo "=== COMPLETE $(date '+%F %T') -> out/${HOST}-appendixD-final.csv ==="
