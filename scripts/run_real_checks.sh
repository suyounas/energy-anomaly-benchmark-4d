#!/usr/bin/env bash
# One-command runner for the real-data checks listed in docs/REVISION_LOG.md.
#   bash scripts/run_real_checks.sh [--dry-run] [--only STEP[,STEP]]
# Steps: tests ablation_gb ablation_ercot tuning_gb tuning_ercot tiers
# Env:   GB_CSV (default neso_demand_full.csv)   ERCOT_CSV (default ercot_demand_full.csv)
#        ERCOT_FLAGS (default ercot_flags.csv; written by code/study4_ercot_eia/02_full_pipeline.py)
#        NEURAL=1 adds lstm_ae and transformer to the equal-budget tuning (slow; GPU advised)
#        N_TRIALS (default 8)  N_SEEDS (default 5)  OUT (default results/real)
# A step whose input file is missing is SKIPPED with a message, not treated as a pass.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}

GB_CSV=${GB_CSV:-neso_demand_full.csv}
ERCOT_CSV=${ERCOT_CSV:-ercot_demand_full.csv}
ERCOT_FLAGS=${ERCOT_FLAGS:-ercot_flags.csv}
OUT=${OUT:-results/real}
N_TRIALS=${N_TRIALS:-8}
N_SEEDS=${N_SEEDS:-5}
FAMS="isolation_forest knn_inductive rolling_z"
[ "${NEURAL:-0}" = "1" ] && FAMS="$FAMS lstm_ae transformer"

DRY=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="$2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac; shift
done

mkdir -p "$OUT"
RAN=""; SKIPPED=""; FAILED=""   # plain strings: macOS bash 3.2 + set -u dislikes empty arrays

want() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; }

step() {  # step NAME REQUIRED_FILE cmd...
  local name=$1 need=$2; shift 2
  want "$name" || return 0
  if [ -n "$need" ] && [ ! -f "$need" ]; then
    echo "[SKIP] $name: missing input $need"; SKIPPED="$SKIPPED $name"; return 0
  fi
  echo "[RUN ] $name"; echo "       $*"
  if [ $DRY -eq 1 ]; then RAN="$RAN $name(dry)"; return 0; fi
  if "$@" 2>&1 | tee "$OUT/$name.log"; then RAN="$RAN $name"; else FAILED="$FAILED $name"; fi
}

GB_EVENTS=(--event 2020-03-23 2020-05-31 COVID --event 2022-12-08 2022-12-17 ColdSnap)
GB_TRAIN=(--train-window 2018-06-01 2020-02-01)
ER_EVENTS=(--event 2021-02-14 2021-02-19 "Texas Freeze" --event 2022-12-22 2022-12-26 "Elliott spike")
ER_TRAIN=(--train-window 2021-06-01 2022-06-01)

step tests "" "$PY" -m pytest -q
step ablation_gb "$GB_CSV" "$PY" experiments/real_resolution_ablation.py --csv "$GB_CSV" \
  --time-col Datetime --value-col ND_MW --base-minutes 30 "${GB_EVENTS[@]}" "${GB_TRAIN[@]}" \
  --factors 1 2 48 --out "$OUT/gb_resolution_ablation.csv"
step ablation_ercot "$ERCOT_CSV" "$PY" experiments/real_resolution_ablation.py --csv "$ERCOT_CSV" \
  --time-col Datetime --value-col Demand_MW --base-minutes 60 "${ER_EVENTS[@]}" "${ER_TRAIN[@]}" \
  --factors 1 24 --out "$OUT/ercot_resolution_ablation.csv"
step tuning_gb "$GB_CSV" "$PY" experiments/equal_budget_tuning.py --csv "$GB_CSV" \
  --time-col Datetime --value-col ND_MW --base-minutes 30 --factor 2 "${GB_EVENTS[@]}" "${GB_TRAIN[@]}" \
  --families $FAMS --n-trials "$N_TRIALS" --n-seeds "$N_SEEDS" --out "$OUT/gb_equal_budget"
step tuning_ercot "$ERCOT_CSV" "$PY" experiments/equal_budget_tuning.py --csv "$ERCOT_CSV" \
  --time-col Datetime --value-col Demand_MW --base-minutes 60 --factor 1 "${ER_EVENTS[@]}" "${ER_TRAIN[@]}" \
  --families $FAMS --n-trials "$N_TRIALS" --n-seeds "$N_SEEDS" --out "$OUT/ercot_equal_budget"
step tiers "$ERCOT_FLAGS" "$PY" experiments/ercot_label_tiers.py --flags "$ERCOT_FLAGS" \
  --out "$OUT/ercot_tiers.csv"

echo; echo "== summary =="
echo "ran:    ${RAN:- none}"
echo "skipped:${SKIPPED:- none}"
echo "failed: ${FAILED:- none}"
[ -z "$FAILED" ]
