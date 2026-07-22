# Author: Riyang
# Stay Point Detection method 8:
# Feature engineering + HistGradientBoostingClassifier (supervised)
# Train on first 80% of agents, predict on remaining 20% (test only).
# Serves as a supervised performance ceiling.

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import pyarrow.parquet as parquet
import os
from multiprocessing import Pool, cpu_count
from utils import print_time as print


def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def compute_features(g):
    """Compute per-point features for one agent's trajectory."""
    n = len(g)
    lats = g["latitude"].values
    lons = g["longitude"].values

    disp_next = np.zeros(n)
    disp_next[:-1] = haversine_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])

    disp_prev = np.zeros(n)
    disp_prev[1:] = disp_next[:-1]

    features = pd.DataFrame(index=g.index)
    features["disp_next"] = disp_next
    features["disp_prev"] = disp_prev

    for w in [5, 15, 30, 60]:
        features[f"lat_std_{w}"] = g["latitude"].rolling(w, center=True, min_periods=1).std()
        features[f"lon_std_{w}"] = g["longitude"].rolling(w, center=True, min_periods=1).std()
        features[f"mean_disp_{w}"] = pd.Series(disp_next, index=g.index).rolling(w, center=True, min_periods=1).mean()
        features[f"max_disp_{w}"] = pd.Series(disp_next, index=g.index).rolling(w, center=True, min_periods=1).max()

        clat = g["latitude"].rolling(w, center=True, min_periods=1).mean()
        clon = g["longitude"].rolling(w, center=True, min_periods=1).mean()
        features[f"dist_centroid_{w}"] = haversine_vec(lats, lons, clat.values, clon.values)

    features["hour"] = g["time"].dt.hour

    return features


def label_points(g, gt_agent):
    """Label each trajectory point as stopped (1) or moving (0)."""
    labels = np.zeros(len(g), dtype=int)
    for _, sp in gt_agent.iterrows():
        mask = (g["time"] >= sp["arrive_time"]) & (g["time"] <= sp["leave_time"])
        labels[mask.values] = 1
    return labels


def segment_predictions(g, preds, min_duration=300):
    """Convert per-point predictions to staypoint intervals."""
    changes = np.diff(preds)
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if preds[0] == 1:
        starts = np.insert(starts, 0, 0)
    if preds[-1] == 1:
        ends = np.append(ends, len(preds))

    staypoints = []
    for s, e in zip(starts, ends):
        seg = g.iloc[s:e]
        duration = (seg["time"].iloc[-1] - seg["time"].iloc[0]).total_seconds()
        if duration >= min_duration:
            staypoints.append({
                "latitude": float(seg["latitude"].mean()),
                "longitude": float(seg["longitude"].mean()),
                "arrive_time": seg["time"].iloc[0],
                "leave_time": seg["time"].iloc[-1],
                "duration_s": float(duration),
                "n_points": int(len(seg)),
            })
    return staypoints


def _build_train_data(args):
    """Build features and labels for one training agent."""
    agent_id, g, gt_agent = args
    g = g.reset_index(drop=True)
    if len(g) < 10 or gt_agent.empty:
        return None, None
    feats = compute_features(g)
    labels = label_points(g, gt_agent)
    pos_mask = labels == 1
    n_pos = pos_mask.sum()
    if n_pos == 0:
        return None, None
    neg_indices = np.where(~pos_mask)[0]
    sampled_neg = np.random.RandomState(42).choice(neg_indices, size=min(n_pos * 2, len(neg_indices)), replace=False)
    keep = np.sort(np.concatenate([np.where(pos_mask)[0], sampled_neg]))
    return feats.iloc[keep], labels[keep]


def _predict_agent(args):
    """Predict staypoints for one agent."""
    agent_id, g, clf, min_duration = args
    g = g.reset_index(drop=True)
    if len(g) < 10:
        return []
    feats = compute_features(g)
    preds = clf.predict(feats)
    sps = segment_predictions(g, preds, min_duration=min_duration)
    for sp in sps:
        sp["agent_id"] = int(agent_id)
    return sps


def get_stay_points(df, min_duration=300, gt_df=None):
    required = {"agent_id", "n_lat", "n_lon", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time", "n_lat", "n_lon", "agent_id"])
    d = d.drop(columns=["latitude", "longitude"], errors="ignore")
    d = d.rename(columns={"n_lat": "latitude", "n_lon": "longitude"})
    d = d.sort_values(["agent_id", "time"]).reset_index(drop=True)

    # Load ground truth for training labels
    if gt_df is None:
        # Fallback: load v1 ground truth directly (backward compatible)
        gt_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "v1", "ground_truth.parquet")
        gt_df = parquet.read_table(gt_path).to_pandas()
        gt_df.rename(columns={"startTime": "arrive_time", "endTime": "leave_time"}, inplace=True)
    gt_df["arrive_time"] = pd.to_datetime(gt_df["arrive_time"])
    gt_df["leave_time"] = pd.to_datetime(gt_df["leave_time"])

    all_agents = sorted(d["agent_id"].unique())

    n_train = max(1, int(len(all_agents) * 0.8))
    train_agents = all_agents[:n_train]
    test_agents = all_agents[n_train:]
    print(f"Training on {len(train_agents)} agents, predicting on {len(test_agents)} test agents")

    # Build training data in parallel
    print(f"Building features and labels with {cpu_count()} workers...")
    train_groups = []
    for agent_id in train_agents:
        g = d[d["agent_id"] == agent_id]
        gt_agent = gt_df[gt_df["agent_id"] == agent_id]
        train_groups.append((agent_id, g, gt_agent))

    with Pool(cpu_count()) as pool:
        train_results = pool.map(_build_train_data, train_groups)

    train_X_list = [x for x, y in train_results if x is not None]
    train_y_list = [y for x, y in train_results if x is not None]

    train_X = pd.concat(train_X_list, ignore_index=True)
    train_y = np.concatenate(train_y_list)
    print(f"Training data: {len(train_X)} samples, {train_y.sum()} positives ({train_y.mean():.2%})")

    # Train classifier (single-threaded, sklearn handles internal parallelism)
    print("Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    clf.fit(train_X, train_y)
    print("Training complete.")

    # Predict on test agents only (serial — clf is large and costly to pickle across workers)
    print(f"Predicting on {len(test_agents)} test agents...")
    out_rows = []
    for agent_id in test_agents:
        g = d[d["agent_id"] == agent_id].reset_index(drop=True)
        if len(g) < 10:
            continue
        feats = compute_features(g)
        preds = clf.predict(feats)
        sps = segment_predictions(g, preds, min_duration=min_duration)
        for sp in sps:
            sp["agent_id"] = int(agent_id)
        out_rows.extend(sps)

    return pd.DataFrame(
        out_rows,
        columns=["agent_id", "latitude", "longitude", "arrive_time", "leave_time", "duration_s", "n_points"]
    )