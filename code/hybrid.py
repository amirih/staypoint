"""
Unsupervised hybrid staypoint detection.

Approach: Confirmation filter.
  Take a primary algorithm's staypoints unchanged (no averaging, no drift),
  then keep only those confirmed by at least min_confirmations other algorithms.
  Requiring agreement filters out the primary's false positives, raising precision
  above what any single algorithm achieves while keeping the primary's exact
  time/location boundaries so eval matching is not hurt by averaging.

Usage
-----
    python code/hybrid.py                                    # default: trackintel confirmed by >=1
    python code/hybrid.py --primary trackintel --min_conf 2 # confirmed by >=2 others
    python code/hybrid.py --sweep --save                    # run all variants and save best
"""

import argparse
import os
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "v1")
NL25_DIR     = os.path.join(DATA_DIR, "detected_staypoints_noiselevel25_dropoutlevel2")

ALGO_DIRS = {
    "hsw":        "algorithm_hsw",
    "hmm_gem":    "algorithm_hmm-gem",
    "dbscan":     "temporal-dbscan",
    "trackintel": "Trackintel_Hyperband_Search",
}

sys.path.insert(0, SCRIPT_DIR)
import eval as eval_utils


# ── data loading ──────────────────────────────────────────────────────────────

def _strip_tz(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        if hasattr(series.dt, "tz") and series.dt.tz is not None:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series


def load_algo(name: str) -> pd.DataFrame:
    path = os.path.join(NL25_DIR, ALGO_DIRS[name], "staypoints.parquet")
    df = pq.read_table(path).to_pandas()
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "pred_lat": "latitude", "pred_lon": "longitude",
        "pred_start": "arrive_time", "pred_end": "leave_time",
        "startTime": "arrive_time", "endTime": "leave_time",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    df["arrive_time"] = _strip_tz(pd.to_datetime(df["arrive_time"], errors="coerce"))
    df["leave_time"]  = _strip_tz(pd.to_datetime(df["leave_time"],  errors="coerce"))
    df = df.dropna(subset=["arrive_time", "leave_time", "latitude", "longitude"])
    return df[["agent_id", "latitude", "longitude", "arrive_time", "leave_time"]].copy()


# ── confirmation filter ───────────────────────────────────────────────────────

def confirmation_filter(primary_df: pd.DataFrame, others: dict,
                        min_confirmations: int = 1,
                        r: float = 0.001, t_min: float = 5.0) -> pd.DataFrame:
    """
    Keep only rows of primary_df that are confirmed by at least
    min_confirmations algorithms in `others`.

    Uses the same spatial (r degrees) and temporal (t_min minutes) matching
    criteria as eval.py so a confirmation means genuine agreement.
    """
    t = pd.Timedelta(minutes=t_min)
    keep = []

    # index each secondary by agent for fast lookup
    by_agent = {name: df.groupby("agent_id") for name, df in others.items()}

    for _, row in primary_df.iterrows():
        agent = row["agent_id"]
        lat, lon = row["latitude"], row["longitude"]
        arr, lv  = row["arrive_time"], row["leave_time"]
        confirmations = 0

        for name, groups in by_agent.items():
            if agent not in groups.groups:
                continue
            ag = groups.get_group(agent)
            match = ag[
                ag["latitude"].between(lat - r, lat + r) &
                ag["longitude"].between(lon - r, lon + r) &
                ag["arrive_time"].between(arr - t, arr + t) &
                ag["leave_time"].between(lv  - t, lv  + t)
            ]
            if not match.empty:
                confirmations += 1
                if confirmations >= min_confirmations:
                    break

        if confirmations >= min_confirmations:
            keep.append(row)

    if not keep:
        return pd.DataFrame(columns=primary_df.columns)
    return pd.DataFrame(keep).reset_index(drop=True)


# ── evaluation ────────────────────────────────────────────────────────────────

def load_gt() -> pd.DataFrame:
    gt = pq.read_table(os.path.join(DATA_DIR, "ground_truth.parquet")).to_pandas()
    gt.rename(columns={"startTime": "arrive_time", "endTime": "leave_time"}, inplace=True)
    gt["arrive_time"] = pd.to_datetime(gt["arrive_time"])
    gt["leave_time"]  = pd.to_datetime(gt["leave_time"])
    return gt


def evaluate(result: pd.DataFrame, gt: pd.DataFrame,
             label: str = "", save_dir: str = None,
             primary: str = "", min_conf: int = 0, others_used: list = None):
    score = eval_utils.get_score(gt, result)
    print(f"  {label:<40}  SP={len(result):>6}  "
          f"F1={score['f1']:.4f}  P={score['precision']:.4f}  R={score['recall']:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result.to_parquet(os.path.join(save_dir, "staypoints.parquet"), index=False)
        result.to_csv(os.path.join(save_dir, "staypoints.csv"), index=False)
        _write_info(save_dir, score, primary, min_conf, others_used, len(result), len(gt))
    return score


def _write_info(out_dir, score, primary, min_conf, others_used, n_pred, n_gt):
    lines = [
        "Method: hybrid (confirmation filter)",
        "",
        "Parameters:",
        f"  primary_algorithm={primary}",
        f"  min_confirmations={min_conf}",
        f"  confirming_algorithms={', '.join(others_used or [])}",
        f"  spatial_radius_deg=0.001",
        f"  temporal_window_min=5",
        f"  noise_level=25",
        f"  dropout_level=2",
        "",
        "Evaluation:",
        f"  f1: {score['f1']:.4f}",
        f"  precision: {score['precision']:.4f}",
        f"  recall: {score['recall']:.4f}",
        f"  predicted_staypoints: {n_pred}",
        f"  ground_truth_staypoints: {n_gt}",
    ]
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── sweep / single run ────────────────────────────────────────────────────────

def run_single(primary: str, min_conf: int, save: bool, gt: pd.DataFrame,
               all_loaded: dict):
    others = {n: df for n, df in all_loaded.items() if n != primary}
    primary_df = all_loaded[primary]

    t0 = time.time()
    result = confirmation_filter(primary_df, others, min_confirmations=min_conf)
    elapsed = time.time() - t0

    label = f"{primary}  confirmed_by>={min_conf}"
    others_used = list(others.keys())

    save_dir = None
    if save:
        folder = f"hybrid_confirm_{primary}_min{min_conf}"
        save_dir = os.path.join(NL25_DIR, folder)

    score = evaluate(result, gt, label=label, save_dir=save_dir,
                     primary=primary, min_conf=min_conf, others_used=others_used)
    return score


def run_sweep(save: bool):
    print("\nLoading ground truth...")
    gt = load_gt()
    print(f"  {len(gt):,} ground truth staypoints\n")

    print("Loading algorithm outputs...")
    all_loaded = {}
    for name in ALGO_DIRS:
        df = load_algo(name)
        all_loaded[name] = df
        print(f"  {name:<12}  {len(df):>6} staypoints")
    print()

    print("Running confirmation filter variants...")
    print(f"  {'Config':<40}  {'SP':>6}  {'F1':>6}  {'P':>6}  {'R':>6}")
    print(f"  {'-'*64}")

    best_score, best_label = None, ""
    # only top-2 as primary; lower-quality algos can still act as confirmers
    TOP_PRIMARIES = ["hsw", "trackintel"]
    for primary in TOP_PRIMARIES:
        n_others = len(ALGO_DIRS) - 1   # 3 potential confirmers
        for min_conf in range(1, n_others + 1):
            score = run_single(primary, min_conf, save, gt, all_loaded)
            if best_score is None or score["precision"] > best_score["precision"]:
                best_score = score
                best_label = f"{primary} min_conf={min_conf}"

    print(f"\n  Best precision: {best_label}  P={best_score['precision']:.4f}  F1={best_score['f1']:.4f}")

    # reference baselines
    print("\nBaseline reference (individual algorithms):")
    for name, df in all_loaded.items():
        score = eval_utils.get_score(gt, df)
        print(f"  {name:<12}  SP={len(df):>6}  "
              f"F1={score['f1']:.4f}  P={score['precision']:.4f}  R={score['recall']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep",    action="store_true",
                        help="Run all primary/min_conf combinations")
    parser.add_argument("--primary",  type=str, default="trackintel",
                        help="Primary algorithm (default: trackintel)")
    parser.add_argument("--min_conf", type=int, default=1,
                        help="Min confirmations from other algorithms (default: 1)")
    parser.add_argument("--save",     action="store_true",
                        help="Save parquet/csv/info.txt for each run")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(save=args.save)
    else:
        gt = load_gt()
        all_loaded = {n: load_algo(n) for n in ALGO_DIRS}
        print(f"\nGround truth: {len(gt):,} staypoints")
        for n, df in all_loaded.items():
            print(f"  Loaded {n:<12}  {len(df):>6} staypoints")
        print()
        run_single(args.primary, args.min_conf, args.save, gt, all_loaded)
