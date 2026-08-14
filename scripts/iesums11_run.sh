#!/usr/bin/env bash
set -euo pipefail

HOST="$(hostname -s)"
LOGDIR="out/campaign_${HOST}"
mkdir -p "$LOGDIR" out logs

TABLE_GAMMAS=(3 5 10)

echo "=== Appendix D / $HOST / started $(date '+%F %T') ==="

for T in 50 75 100; do
  for G in "${TABLE_GAMMAS[@]}"; do
    echo "[$(date '+%F %T')] START  table_T${T}_G${G}_omip_cct"
    python3 run_experiments.py \
      --set_name timestudyfixedzero --solvers OMIP,CCTA --perm none \
      --gamma "$G" --T "$T" \
      >"$LOGDIR/table_T${T}_G${G}_omip_cct.log" 2>&1
    echo "[$(date '+%F %T')] DONE   table_T${T}_G${G}_omip_cct"
  done
done

for G in "${TABLE_GAMMAS[@]}"; do
  echo "[$(date '+%F %T')] START  table_T200_G${G}_omip_cct"
  python3 run_experiments.py \
    --set_name timestudyfixedzero --solvers OMIP,CCTA --perm none \
    --gamma "$G" --T 200 \
    >"$LOGDIR/table_T200_G${G}_omip_cct.log" 2>&1
  echo "[$(date '+%F %T')] DONE   table_T200_G${G}_omip_cct"
done

mv out/final.csv "out/${HOST}-appendixD-final.csv"
echo "=== COMPLETE $(date '+%F %T') -> out/${HOST}-appendixD-final.csv ==="
