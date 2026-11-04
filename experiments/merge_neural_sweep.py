"""Merge the neural-only sweep CSV into the classical sweep CSV and refresh the summary and figures.

  python3 experiments/merge_neural_sweep.py --kind shift     # reads results/synthetic/sweep_shift_mag3.csv and sweep_shift_mag3_neural.csv
"""
import argparse, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analyse_sweep import summarise

ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=["shift", "burst"])
ap.add_argument("--dir", default="results/synthetic")
a = ap.parse_args()
base_p = f"{a.dir}/sweep_{a.kind}_mag3.csv"; neu_p = f"{a.dir}/sweep_{a.kind}_mag3_neural.csv"
base = pd.read_csv(base_p); neu = pd.read_csv(neu_p)
assert set(neu.detector) <= {"lstm_ae", "transformer"}, set(neu.detector)
base = base[~base.detector.isin(["lstm_ae", "transformer"])]
merged = pd.concat([base, neu], ignore_index=True)
merged.to_csv(base_p, index=False)
S = summarise(merged); S.to_csv(f"{a.dir}/summary_{a.kind}.csv", index=False)
print(f"{a.kind}: {len(neu)} neural rows merged; detectors now:", sorted(set(S.detector)))
print("seeds per neural cell:", S[S.detector.isin(["lstm_ae", "transformer"])].n_seeds.describe()[["min", "max"]].to_dict())
