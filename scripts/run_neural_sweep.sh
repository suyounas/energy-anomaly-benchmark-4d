#!/usr/bin/env bash
# Neural (LSTM-AE, Transformer) arms of the controlled sweep, resumable per seed.  Run on a GPU machine (Colab: Runtime > Change runtime type > GPU).
#
#   PROFILE=quick bash scripts/run_neural_sweep.sh     # 3 seeds, batch 128, 20 epochs: smoke run, roughly 20-40 min on a T4
#   PROFILE=full  bash scripts/run_neural_sweep.sh     # 10 seeds, batch 128, 20 epochs: the run the paper needs, roughly 1-2 h per event type on a T4
#   PROFILE=paper bash scripts/run_neural_sweep.sh     # 10 seeds, batch 32, 30 epochs (exactly Table 'det'); expect several hours per event type
#
# Each seed is written to results/synthetic/neural_parts/<kind>_seed<k>.csv, so a disconnected session resumes where it stopped (re-run the same command).
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${PROFILE:-full}"
case "$PROFILE" in
  quick) SEEDS=3;  BATCH=128; EPOCHS=20 ;;
  full)  SEEDS=10; BATCH=128; EPOCHS=20 ;;
  paper) SEEDS=10; BATCH=32;  EPOCHS=30 ;;
  *) echo "PROFILE must be quick, full or paper"; exit 1 ;;
esac
python3 - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else "No GPU found: switch the runtime to a GPU before running (CPU would take days).")
PY
mkdir -p results/synthetic/neural_parts
for KIND in shift burst; do
  for ((S=0; S<SEEDS; S++)); do
    OUT="results/synthetic/neural_parts/${KIND}_seed${S}.csv"
    if [[ -s "$OUT" ]]; then echo "skip $OUT (done)"; continue; fi
    echo "== $KIND seed $S  ($(date +%H:%M:%S))"
    python3 experiments/synthetic_sweep.py --kind "$KIND" --neural-only --seeds 1 --seed-start "$S" \
        --neural-batch "$BATCH" --neural-epochs "$EPOCHS" --out "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
  done
  python3 - "$KIND" "$PROFILE" <<'PY'
import sys, glob, pandas as pd
kind, profile = sys.argv[1], sys.argv[2]
parts = sorted(glob.glob(f"results/synthetic/neural_parts/{kind}_seed*.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df.to_csv(f"results/synthetic/sweep_{kind}_mag3_neural.csv", index=False)
print(kind, "seeds:", df.seed.nunique(), "rows:", len(df))
open(f"results/synthetic/neural_profile_{kind}.txt", "w").write(profile + "\n")
PY
  python3 experiments/merge_neural_sweep.py --kind "$KIND"
done
echo "Done. Next (any machine with the repo):"
echo "  python3 paper/revised/make_tables.py && python3 paper/revised/make_figures.py && python3 -m pytest -q"
echo "  then send the results/synthetic/*.csv files back, or commit them (git add -f results/synthetic)."
