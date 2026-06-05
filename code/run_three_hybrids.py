"""
Build three hybrid staypoint detectors across all 16 noise/dropout combinations.

Hybrid 1 — Unsupervised
  Primary:    HMM-GEM  (best single-shot unsupervised w/ staypoints)
  Confirmers: HSW, T-DBSCAN  (single-shot unsupervised only, P > 0.5)
  Output:     hybrid_unsupervised/

Hybrid 2 — Optimized
  Primary:    SSPE (grid-search / parameter-optimized)
  Confirmers: HMM-GEM, Trackintel Hyperband, HSW, T-DBSCAN sweep, ASW
              (all non-baseline, non-GB; optimized versions replace single-shot)
  Output:     hybrid_optimized/

Hybrid 3 — Fully Supervised
  Primary:    SSPE (same as optimized)
  Confirmers: same as optimized + Gradient Boosting
  Output:     hybrid_supervised/

All hybrids use T=0.90 (logit-weight threshold) — any single confirmer suffices.
Weights = logit(precision) using NL25/DL2 evaluation as reference.
Algorithms with precision <= 0.5 excluded (negative logit, meaningless).
"""

import json
import math
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "v1")

sys.path.insert(0, SCRIPT_DIR)
import eval as eval_utils

NOISE_LEVELS   = [0, 10, 25, 50]
DROPOUT_LEVELS = [0, 1, 2, 3]
THRESHOLD      = 0.90

# Precision values from NL25/DL2 evaluation.json (used as fixed weights everywhere)
PRECISION = {
    "algorithm_hmm-gem":            0.7940,
    "algorithm_hsw":                0.7850,
    "temporal-dbscan":              0.7117,
    "temporal-dbscan-sweep":        0.7733,
    "Trackintel_Hyperband_Search":  0.8098,
    "asw":                          0.6218,
    "algorithm_gradient_boosting":  0.8407,
    "sspe":                         0.8348,
}

# Three hybrid configurations
HYBRIDS = {
    "hybrid_unsupervised": {
        "primary":    "algorithm_hmm-gem",
        "confirmers": ["algorithm_hsw", "temporal-dbscan"],
        # centroid_sliding_window (P=0.4875) excluded — below random, negative logit
    },
    "hybrid_optimized": {
        "primary":    "sspe",
        "confirmers": [
            "algorithm_hmm-gem",
            "Trackintel_Hyperband_Search",
            "algorithm_hsw",
            "temporal-dbscan-sweep",
            "asw",           # P=0.622, logit=0.498 — low but positive
            # centroid (P=0.4875) excluded
            # temporal-dbscan replaced by sweep
            # gradient_boosting excluded for optimized
        ],
    },
    "hybrid_supervised": {
        "primary":    "sspe",
        "confirmers": [
            "algorithm_hmm-gem",
            "Trackintel_Hyperband_Search",
            "algorithm_hsw",
            "temporal-dbscan-sweep",
            "asw",
            "algorithm_gradient_boosting",   # added for fully supervised
        ],
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_tz(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def _load(nl_dir, folder):
    """Load staypoints.parquet (or staypoint.parquet for asw)."""
    for name in ["staypoints.parquet", "staypoint.parquet"]:
        path = os.path.join(nl_dir, folder, name)
        if os.path.exists(path):
            df = pq.read_table(path).to_pandas()
            df.columns = [c.strip() for c in df.columns]
            for old, new in {"startTime": "arrive_time", "endTime": "leave_time",
                             "pred_start": "arrive_time", "pred_end": "leave_time"}.items():
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)
            df["arrive_time"] = _strip_tz(pd.to_datetime(df["arrive_time"], errors="coerce"))
            df["leave_time"]  = _strip_tz(pd.to_datetime(df["leave_time"],  errors="coerce"))
            return df.dropna(subset=["arrive_time", "leave_time", "latitude", "longitude"])[
                ["agent_id", "latitude", "longitude", "arrive_time", "leave_time"]].copy()
    return None


def _confirm(primary, confirmers_w, T, r=0.001, t_min=5.0):
    t = pd.Timedelta(minutes=t_min)
    by_agent = {n: (w, df.groupby("agent_id")) for n, (df, w) in confirmers_w.items()}
    keep = []
    for _, row in primary.iterrows():
        ag, lat, lon = row["agent_id"], row["latitude"], row["longitude"]
        arr, lv = row["arrive_time"], row["leave_time"]
        score = 0.0
        for n, (w, grps) in by_agent.items():
            if ag not in grps.groups:
                continue
            g = grps.get_group(ag)
            if not g[
                g["latitude"].between(lat - r, lat + r) &
                g["longitude"].between(lon - r, lon + r) &
                g["arrive_time"].between(arr - t, arr + t) &
                g["leave_time"].between(lv - t, lv + t)
            ].empty:
                score += w
        if score >= T:
            keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True) if keep else pd.DataFrame(
        columns=primary.columns)


def _save(out_dir, result, score_basic, score_multi, hybrid_name, primary_name,
          confirmers_info, gt_n, T):
    os.makedirs(out_dir, exist_ok=True)
    result.to_parquet(os.path.join(out_dir, "staypoints.parquet"), index=False)

    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(score_basic, f, indent=2)
    with open(os.path.join(out_dir, "eval_multi_rt.json"), "w") as f:
        json.dump(score_multi, f, indent=2)

    lines = [
        f"Method: {hybrid_name} (logit-weighted confirmation filter)",
        "",
        "Parameters:",
        f"  primary={primary_name}",
        f"  threshold={T}",
        "  confirmers (name: precision -> logit_weight):",
    ]
    for name, info in confirmers_info.items():
        lines.append(f"    {name}: {info['p']:.3f} -> {info['w']:.3f}")
    lines += [
        "",
        "Evaluation (r=0.001, t=5):",
        f"  f1: {score_basic['f1']:.4f}",
        f"  precision: {score_basic['precision']:.4f}",
        f"  recall: {score_basic['recall']:.4f}",
        f"  predicted_staypoints: {len(result)}",
        f"  ground_truth_staypoints: {gt_n}",
    ]
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── per-combo runner ──────────────────────────────────────────────────────────

def run_combo(noise, dropout, gt):
    nl_dir = os.path.join(DATA_DIR,
                          f"detected_staypoints_noiselevel{noise}_dropoutlevel{dropout}")
    if not os.path.isdir(nl_dir):
        return {}

    scores = {}
    for hybrid_name, cfg in HYBRIDS.items():
        primary_df = _load(nl_dir, cfg["primary"])
        if primary_df is None:
            print(f"    {hybrid_name}: SKIP (primary {cfg['primary']} missing)")
            continue

        confirmers_w    = {}
        confirmers_info = {}
        for folder in cfg["confirmers"]:
            df = _load(nl_dir, folder)
            if df is None:
                continue
            p = PRECISION.get(folder, 0.5)
            w = math.log(p / (1 - p))
            confirmers_w[folder]    = (df, w)
            confirmers_info[folder] = {"p": p, "w": w}

        if not confirmers_w:
            print(f"    {hybrid_name}: SKIP (no confirmers)")
            continue

        result = _confirm(primary_df, confirmers_w, THRESHOLD)
        if result.empty:
            print(f"    {hybrid_name}: SKIP (no staypoints produced)")
            continue

        score_basic = eval_utils.get_score(gt, result, is_f1_only=False)
        score_multi = eval_utils.get_multi_f1_score(gt, result)

        out_dir = os.path.join(nl_dir, hybrid_name)
        _save(out_dir, result, score_basic, score_multi,
              hybrid_name, cfg["primary"], confirmers_info, len(gt), THRESHOLD)

        scores[hybrid_name] = score_basic
        print(f"    {hybrid_name:<22} {len(primary_df):>6}->{len(result):>6} SP  "
              f"F1={score_basic['f1']:.4f}  P={score_basic['precision']:.4f}  "
              f"R={score_basic['recall']:.4f}")

    return scores


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    gt = pq.read_table(os.path.join(DATA_DIR, "ground_truth.parquet")).to_pandas()
    gt.rename(columns={"startTime": "arrive_time", "endTime": "leave_time"}, inplace=True)
    gt["arrive_time"] = pd.to_datetime(gt["arrive_time"])
    gt["leave_time"]  = pd.to_datetime(gt["leave_time"])
    print(f"Ground truth: {len(gt):,} staypoints\n")

    all_scores = {}
    for noise in NOISE_LEVELS:
        for dropout in DROPOUT_LEVELS:
            print(f"NL={noise} DL={dropout}")
            scores = run_combo(noise, dropout, gt)
            all_scores[(noise, dropout)] = scores
            print()

    # summary
    print("=" * 80)
    print(f"SUMMARY  (T={THRESHOLD})")
    print("=" * 80)
    for hybrid_name in HYBRIDS:
        print(f"\n  {hybrid_name}")
        print(f"  {'NL':>4} {'DL':>4}  {'F1':>7}  {'P':>8}  {'R':>7}")
        print(f"  {'-'*38}")
        for noise in NOISE_LEVELS:
            for dropout in DROPOUT_LEVELS:
                s = all_scores.get((noise, dropout), {}).get(hybrid_name)
                if s:
                    print(f"  {noise:>4} {dropout:>4}  {s['f1']:>7.4f}  "
                          f"{s['precision']:>8.4f}  {s['recall']:>7.4f}")


if __name__ == "__main__":
    main()
