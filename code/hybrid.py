"""
Hybrid staypoint detection — weighted confirmation filter.

Primary algorithm: HMM-GEM (best F1 among algorithms with staypoints available).
Confirmers: all quality-gated algorithms (precision > 0.70), weighted by logit(precision).

A staypoint from the primary is kept when:
    Σ logit(precision_i) for each confirmer that agrees  ≥  threshold

Usage
-----
    python code/hybrid.py                      # threshold sweep, pick best
    python code/hybrid.py --threshold 1.3      # single run
    python code/hybrid.py --save --threshold 1.3
"""

import argparse
import json
import math
import os
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "v1")
NL25_DIR     = os.path.join(DATA_DIR, "detected_staypoints_noiselevel25_dropoutlevel2")

sys.path.insert(0, SCRIPT_DIR)
import eval as eval_utils

# ── algorithm registry ────────────────────────────────────────────────────────
# precision values from evaluation.json at r=0.001, t=5 on NL25/DL2

PRIMARY = {
    "name": "hmm_gem",
    "path": os.path.join(NL25_DIR, "algorithm_hmm-gem", "staypoints.parquet"),
    "precision": 0.7940,
}

# Quality gate: only algorithms with precision > 0.70 and staypoints.parquet
CONFIRMERS = {
    "gradient_boosting": {
        "path": os.path.join(NL25_DIR, "algorithm_gradient_boosting", "staypoints.parquet"),
        "precision": 0.8407,
    },
    "trackintel": {
        "path": os.path.join(NL25_DIR, "Trackintel_Hyperband_Search", "staypoints.parquet"),
        "precision": 0.8098,
    },
    "hsw": {
        "path": os.path.join(NL25_DIR, "algorithm_hsw", "staypoints.parquet"),
        "precision": 0.7850,
    },
    "dbscan_sweep": {
        "path": os.path.join(NL25_DIR, "temporal-dbscan-sweep", "staypoints.parquet"),
        "precision": 0.7733,
    },
    "temporal_dbscan": {
        "path": os.path.join(NL25_DIR, "temporal-dbscan", "staypoints.parquet"),
        "precision": 0.7117,
    },
}

# excluded: sspe (no staypoints.parquet), asw (P=0.622), centroid_sw (P=0.488)


def _logit(p):
    return math.log(p / (1.0 - p))


def _strip_tz(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        if hasattr(series.dt, "tz") and series.dt.tz is not None:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series


def _load(path):
    df = pq.read_table(path).to_pandas()
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "startTime": "arrive_time", "endTime": "leave_time",
        "pred_lat": "latitude", "pred_lon": "longitude",
        "pred_start": "arrive_time", "pred_end": "leave_time",
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    df["arrive_time"] = _strip_tz(pd.to_datetime(df["arrive_time"], errors="coerce"))
    df["leave_time"]  = _strip_tz(pd.to_datetime(df["leave_time"],  errors="coerce"))
    return df.dropna(subset=["arrive_time", "leave_time", "latitude", "longitude"])[
        ["agent_id", "latitude", "longitude", "arrive_time", "leave_time"]
    ].copy()


# ── weighted confirmation filter ──────────────────────────────────────────────

def weighted_confirmation_filter(primary_df, confirmers_loaded, threshold,
                                  r=0.001, t_min=5.0):
    """
    For each staypoint in primary_df, sum the logit(precision) weights of
    every confirmer that has a matching staypoint. Keep if sum >= threshold.
    """
    t = pd.Timedelta(minutes=t_min)

    # pre-group each confirmer by agent for fast lookup
    by_agent = {
        name: (info["weight"], df.groupby("agent_id"))
        for name, (df, info) in confirmers_loaded.items()
    }

    keep = []
    for _, row in primary_df.iterrows():
        agent = row["agent_id"]
        lat, lon = row["latitude"], row["longitude"]
        arr, lv  = row["arrive_time"], row["leave_time"]
        score = 0.0

        for name, (weight, groups) in by_agent.items():
            if agent not in groups.groups:
                continue
            ag = groups.get_group(agent)
            if not ag[
                ag["latitude"].between(lat - r, lat + r) &
                ag["longitude"].between(lon - r, lon + r) &
                ag["arrive_time"].between(arr - t, arr + t) &
                ag["leave_time"].between(lv - t, lv + t)
            ].empty:
                score += weight

        if score >= threshold:
            keep.append(row)

    return pd.DataFrame(keep).reset_index(drop=True) if keep else pd.DataFrame(
        columns=primary_df.columns)


# ── evaluation ────────────────────────────────────────────────────────────────

def load_gt():
    gt = pq.read_table(os.path.join(DATA_DIR, "ground_truth.parquet")).to_pandas()
    gt.rename(columns={"startTime": "arrive_time", "endTime": "leave_time"}, inplace=True)
    gt["arrive_time"] = pd.to_datetime(gt["arrive_time"])
    gt["leave_time"]  = pd.to_datetime(gt["leave_time"])
    return gt


def score_result(result, gt):
    return eval_utils.get_score(gt, result, is_f1_only=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="Logit-weight threshold. Omit to run sweep.")
    parser.add_argument("--save", action="store_true",
                        help="Save best result to hybrid_weighted/ folder")
    args = parser.parse_args()

    print("\nLoading ground truth...")
    gt = load_gt()
    print(f"  {len(gt):,} ground truth staypoints")

    print("\nLoading primary (HMM-GEM)...")
    primary_df = _load(PRIMARY["path"])
    print(f"  {len(primary_df):,} staypoints")

    print("\nLoading confirmers...")
    confirmers_loaded = {}
    for name, info in CONFIRMERS.items():
        df = _load(info["path"])
        w  = _logit(info["precision"])
        confirmers_loaded[name] = (df, {"weight": w, "precision": info["precision"]})
        print(f"  {name:<20}  {len(df):>6} SP  P={info['precision']:.3f}  "
              f"logit_w={w:.3f}")

    total_max_weight = sum(info["weight"] for _, info in confirmers_loaded.values())
    print(f"\n  Max possible score per staypoint: {total_max_weight:.3f}")

    # threshold sweep or single run
    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        # sweep covering: any 1 weak → 1 strong → 2 medium → GB alone → 2 strong
        thresholds = [0.90, 1.22, 1.30, 1.45, 1.67, 2.20, 2.75, 3.50]

    print(f"\n{'Threshold':>10}  {'SPs':>7}  {'F1':>7}  {'Precision':>10}  {'Recall':>8}")
    print("-" * 52)

    best = None
    results_by_threshold = {}
    for T in thresholds:
        t0 = time.time()
        result = weighted_confirmation_filter(primary_df, confirmers_loaded, T)
        elapsed = time.time() - t0
        if result.empty:
            print(f"  T={T:>6.2f}  no staypoints produced")
            continue
        score = score_result(result, gt)
        results_by_threshold[T] = (result, score)
        print(f"  T={T:>6.2f}  {len(result):>7}  "
              f"{score['f1']:>7.4f}  {score['precision']:>10.4f}  "
              f"{score['recall']:>8.4f}  ({elapsed:.1f}s)")

        if best is None or score["f1"] > best[1]["f1"]:
            best = (T, score, result)

    if best:
        T_best, score_best, result_best = best
        print(f"\n  Best F1 at threshold={T_best}: "
              f"F1={score_best['f1']:.4f}  P={score_best['precision']:.4f}  "
              f"R={score_best['recall']:.4f}")

        if args.save:
            out_dir = os.path.join(NL25_DIR, f"hybrid_weighted_T{T_best}")
            os.makedirs(out_dir, exist_ok=True)
            result_best.to_parquet(os.path.join(out_dir, "staypoints.parquet"), index=False)
            result_best.to_csv(os.path.join(out_dir, "staypoints.csv"), index=False)

            info_lines = [
                "Method: hybrid (logit-weighted confirmation filter)",
                "",
                "Parameters:",
                f"  primary=hmm_gem",
                f"  threshold={T_best}",
                f"  confirmers (name: precision -> logit_weight):",
            ]
            for name, (_, info) in confirmers_loaded.items():
                info_lines.append(
                    f"    {name}: {info['precision']:.3f} -> {info['weight']:.3f}")
            info_lines += [
                "",
                "Evaluation (r=0.001, t=5):",
                f"  f1: {score_best['f1']:.4f}",
                f"  precision: {score_best['precision']:.4f}",
                f"  recall: {score_best['recall']:.4f}",
                f"  predicted_staypoints: {len(result_best)}",
                f"  ground_truth_staypoints: {len(gt)}",
            ]
            with open(os.path.join(out_dir, "info.txt"), "w") as f:
                f.write("\n".join(info_lines) + "\n")
            print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    main()
