"""
Run every individual staypoint detection method on a given noise/dropout level,
save parquet results and F1 score JSON files, then print a summary table.

Usage (run from project root):
    python code/run_individual.py                        # noise=0, dropout=0
    python code/run_individual.py --noise 10 --dropout 1

Methods run: hsw, sw, b3_adaptive, m7, m8, alex
Skipped:     ti_baseline (weak baseline), lance (not likely to compete)
"""

import argparse
import json
import os
import sys
import time
import warnings

import pandas as pd
import pyarrow.parquet as pq

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "v1")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "approaches"))

import eval as eval_utils


# ── data helpers ──────────────────────────────────────────────────────────────

def load_traj(noise, dropout):
    path = os.path.join(DATA_DIR,
                        f"trajectories_noiselevel{noise}_dropoutlevel{dropout}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    df = pq.read_table(path).to_pandas()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude", "agent_id"])
    df = df.sort_values(["agent_id", "time"]).reset_index(drop=True)
    # Alias expected by methods that still use the old n_lat/n_lon column names
    df["n_lat"] = df["latitude"]
    df["n_lon"] = df["longitude"]
    return df


def load_gt():
    gt = pq.read_table(os.path.join(DATA_DIR, "ground_truth.parquet")).to_pandas()
    gt.rename(columns={"startTime": "arrive_time", "endTime": "leave_time"}, inplace=True)
    return gt


# ── result persistence ────────────────────────────────────────────────────────

def _result_paths(method_name, params_str, noise, dropout):
    out_dir = os.path.join(
        DATA_DIR,
        f"detected_staypoints_noiselevel{noise}_dropoutlevel{dropout}",
        f"algorithm_{method_name}",
    )
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"{params_str}.parquet")
    score_path   = parquet_path + "_score.json"
    return parquet_path, score_path


def _load_cached_score(score_path):
    if os.path.exists(score_path):
        with open(score_path) as f:
            return json.load(f)
    return None


def _save_score(score, score_path):
    with open(score_path, "w") as f:
        json.dump(score, f, indent=4)


# ── method wrappers ───────────────────────────────────────────────────────────

def run_hsw(df, **kwargs):
    from approaches.hossein import hsw
    from approaches.master import get_stay_points
    return get_stay_points(
        func=hsw,
        df=df[["agent_id", "latitude", "longitude", "time"]],
        **kwargs,
    )


def run_sw(df, **kwargs):
    from approaches.original_sw import sw
    from approaches.master import get_stay_points
    return get_stay_points(
        func=sw,
        df=df[["agent_id", "latitude", "longitude", "time"]],
        **kwargs,
    )


def run_b3_adaptive(df, **kwargs):
    from approaches.mo import b3_adaptive
    from approaches.master import get_stay_points
    return get_stay_points(
        func=b3_adaptive,
        df=df[["agent_id", "latitude", "longitude", "time"]],
        **kwargs,
    )


def run_m7(df, **kwargs):
    from approaches import m7
    return m7.get_stay_points(df, **kwargs)


def run_m8(df, **kwargs):
    from approaches import m8
    return m8.get_stay_points(df, **kwargs)


def run_alex(df, **kwargs):
    from approaches.alex import StayPointDetector
    detector = StayPointDetector(**{
        k: v for k, v in kwargs.items()
        if k in ("eps_meters", "min_time_minutes", "max_gap_minutes")
    })
    result = detector.fit_predict(df)
    if result.empty:
        return pd.DataFrame(columns=[
            "agent_id", "latitude", "longitude",
            "arrive_time", "leave_time", "duration_s", "n_points",
        ])
    result = result.rename(columns={
        "pred_start": "arrive_time",
        "pred_end":   "leave_time",
        "pred_lat":   "latitude",
        "pred_lon":   "longitude",
        "duration":   "duration_s",
    })
    if "n_points" not in result.columns:
        result["n_points"] = 0
    return result[["agent_id", "latitude", "longitude",
                   "arrive_time", "leave_time", "duration_s", "n_points"]]


# ── method registry ───────────────────────────────────────────────────────────
# (name, runner_fn, params_label, kwargs)
METHODS = [
    ("hsw",         run_hsw,        "time5min_dist50m",   {"time_thresh_min": 5,  "dist_thresh_m": 50}),
    ("sw",          run_sw,         "time5min_dist50m",   {"time_thresh_min": 5,  "dist_thresh_m": 50}),
    ("b3_adaptive", run_b3_adaptive,"time20min_dist50m",  {"time_thresh_min": 20, "dist_thresh_m": 50}),
    ("m7",          run_m7,         "default",            {}),
    ("m8",          run_m8,         "default",            {}),
    ("alex",        run_alex,       "eps40m_time5min",    {"eps_meters": 40, "min_time_minutes": 5}),
]


# ── core runner ───────────────────────────────────────────────────────────────

def run_method(name, fn, params_str, kwargs, df, gt, noise, dropout):
    parquet_path, score_path = _result_paths(name, params_str, noise, dropout)

    cached = _load_cached_score(score_path)
    if cached:
        print(f"  [cached]  {name:<15}  F1={cached['f1']:.4f}  "
              f"P={cached['precision']:.4f}  R={cached['recall']:.4f}")
        return cached

    print(f"  Running   {name}...", flush=True)
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fn(df, **kwargs)
    except Exception as exc:
        print(f"  ERROR in {name}: {exc}")
        return None
    elapsed = time.time() - t0

    result.to_parquet(parquet_path, index=False)

    # For M8: only score against the agents it actually predicted (avoids recall
    # being crushed by the 80% training agents that have no predictions).
    gt_for_score = gt
    if name == "m8" and not result.empty:
        predicted_agents = result["agent_id"].unique()
        gt_for_score = gt[gt["agent_id"].isin(predicted_agents)]

    score = eval_utils.get_score(gt_for_score, result)
    _save_score(score, score_path)

    print(f"  Done      {name:<15}  {len(result):>6} staypoints  "
          f"{elapsed:>6.1f}s  "
          f"F1={score['f1']:.4f}  P={score['precision']:.4f}  R={score['recall']:.4f}")
    return score


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise",   type=int, default=0)
    parser.add_argument("--dropout", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Running all methods  |  noise={args.noise}  dropout={args.dropout}")
    print(f"{'='*60}\n")

    df = load_traj(args.noise, args.dropout)
    gt = load_gt()
    print(f"  Trajectory: {len(df):,} points, {df['agent_id'].nunique()} agents")
    print(f"  Ground truth: {len(gt):,} staypoints\n")

    scores = {}
    for name, fn, params_str, kwargs in METHODS:
        score = run_method(name, fn, params_str, kwargs, df, gt, args.noise, args.dropout)
        scores[name] = score

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY  |  noise={args.noise}  dropout={args.dropout}")
    print(f"{'='*60}")
    print(f"  {'Method':<16} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*46}")
    for name, score in scores.items():
        if score:
            print(f"  {name:<16} {score['f1']:>8.4f} {score['precision']:>10.4f} {score['recall']:>8.4f}")
        else:
            print(f"  {name:<16} {'ERROR':>8}")
    print()

    # Save summary JSON
    summary_path = os.path.join(
        DATA_DIR,
        f"summary_noiselevel{args.noise}_dropoutlevel{args.dropout}.json",
    )
    with open(summary_path, "w") as f:
        json.dump({k: v for k, v in scores.items() if v}, f, indent=4)
    print(f"  Summary saved to {summary_path}\n")


if __name__ == "__main__":
    main()
