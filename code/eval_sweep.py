"""
Evaluate selected staypoint methods across the S/T parameter grid from the paper guidelines.

S/T grid:
  Default:  S=0.001, T=5
  S=0.001,  T=1
  S=0.001,  T=10
  S=0.0001, T=5
  S=0.005,  T=5

Usage:
    python code/eval_sweep.py
"""

import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "v1")
NL25_DIR     = os.path.join(DATA_DIR, "detected_staypoints_noiselevel25_dropoutlevel2")

sys.path.insert(0, SCRIPT_DIR)
import eval as eval_utils

# ── S/T grid ──────────────────────────────────────────────────────────────────
ST_GRID = [
    {"s": 0.001,  "t": 5,  "label": "S=0.001 T=5  (default)"},
    {"s": 0.001,  "t": 1,  "label": "S=0.001 T=1"},
    {"s": 0.001,  "t": 10, "label": "S=0.001 T=10"},
    {"s": 0.0001, "t": 5,  "label": "S=0.0001 T=5"},
    {"s": 0.005,  "t": 5,  "label": "S=0.005 T=5"},
]

# ── methods to evaluate ───────────────────────────────────────────────────────
# (label, parquet_path)
METHODS = [
    # best individual baseline for comparison
    ("trackintel",
     os.path.join(NL25_DIR, "Trackintel_Hyperband_Search", "staypoints.parquet")),
    # best hybrid: highest F1 (0.784) and precision (0.922)
    ("hybrid: trackintel conf>=2",
     os.path.join(NL25_DIR, "hybrid_confirm_trackintel_min2", "staypoints.parquet")),
    # runner-up hybrid: best recall balance (F1=0.773, P=0.877)
    ("hybrid: trackintel conf>=1",
     os.path.join(NL25_DIR, "hybrid_confirm_trackintel_min1", "staypoints.parquet")),
]


def _strip_tz(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        if hasattr(series.dt, "tz") and series.dt.tz is not None:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series


def load_parquet(path):
    df = pq.read_table(path).to_pandas()
    df.columns = [c.strip() for c in df.columns]
    rename = {"startTime": "arrive_time", "endTime": "leave_time"}
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    df["arrive_time"] = _strip_tz(pd.to_datetime(df["arrive_time"], errors="coerce"))
    df["leave_time"]  = _strip_tz(pd.to_datetime(df["leave_time"],  errors="coerce"))
    return df


def main():
    # load ground truth
    gt = load_parquet(os.path.join(DATA_DIR, "ground_truth.parquet"))
    print(f"Ground truth: {len(gt):,} staypoints\n")

    # load all method outputs
    loaded = []
    for label, path in METHODS:
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        df = load_parquet(path)
        loaded.append((label, df))
        print(f"  Loaded {label}  ({len(df):,} SP)")
    print()

    # print results table
    col_w = 42
    header = f"{'Method':<{col_w}}" + "".join(
        f"  {'F1':>6} {'P':>6} {'R':>6}" for _ in ST_GRID
    )
    subheader = f"{'':<{col_w}}" + "".join(
        f"  {p['label']:>20}" for p in ST_GRID
    )
    print(subheader)
    print("-" * (col_w + len(ST_GRID) * 22))

    all_results = {}
    for label, df in loaded:
        all_results[label] = {}
        # compute overlap once per method (independent of S/T)
        overlap = eval_utils.get_overlap_score(gt, df, placeholder=False)

        for params in ST_GRID:
            print(f"  scoring {label[:35]}  {params['label']}...", flush=True)
            score = eval_utils.get_score(gt, df, r=params["s"], t=params["t"])
            score["temporal_overlap_score"]          = overlap["temporal_overlap_score"]
            score["spatial_overlap_score"]           = overlap["spatial_overlap_score"]
            score["spatial_temporal_overlap_score"]  = overlap["spatial_temporal_overlap_score"]
            all_results[label][params["label"]] = score

    # print summary table
    print(f"\n{'Method':<{col_w}}  {'S/T':<20}  {'F1':>6}  {'P':>6}  {'R':>6}  {'TempOvlp':>9}  {'SpatDist':>9}")
    print("-" * (col_w + 70))
    for label, results in all_results.items():
        for st_label, score in results.items():
            print(f"  {label:<{col_w-2}}  {st_label:<20}  "
                  f"{score['f1']:>6.4f}  {score['precision']:>6.4f}  {score['recall']:>6.4f}  "
                  f"{score['temporal_overlap_score']:>9.4f}  {score['spatial_overlap_score']:>9.4f}")
        print()

    # save JSON summary
    out_path = os.path.join(NL25_DIR, "eval_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
