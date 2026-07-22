# Author: Riyang
# Stay Point Detection method 7:
# Hidden Markov Model with 2 states (stopped / moving)
# Manual Viterbi implementation using numpy (no hmmlearn needed).
# Observation: displacement between consecutive points.

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from utils import print_time as print


def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def viterbi_2state(displacements, pi, log_A, means, stds):
    """Viterbi decoding for 2-state Gaussian HMM."""
    n = len(displacements)
    log_emit = np.zeros((n, 2))
    for s in range(2):
        diff = displacements - means[s]
        log_emit[:, s] = -0.5 * (diff / stds[s])**2 - np.log(stds[s] * np.sqrt(2 * np.pi))

    V = np.zeros((n, 2))
    backptr = np.zeros((n, 2), dtype=int)
    V[0] = np.log(pi + 1e-300) + log_emit[0]

    for t in range(1, n):
        for s in range(2):
            trans = V[t - 1] + log_A[:, s]
            backptr[t, s] = np.argmax(trans)
            V[t, s] = np.max(trans) + log_emit[t, s]

    states = np.zeros(n, dtype=int)
    states[-1] = np.argmax(V[-1])
    for t in range(n - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]
    return states


def _process_agent(args):
    agent_id, g, pi, log_A, means, stds, min_duration = args
    g = g.reset_index(drop=True)
    n = len(g)
    if n < 2:
        return []

    lats = g["latitude"].values
    lons = g["longitude"].values

    disps = haversine_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])
    disps = np.append(disps, 0.0)

    states = viterbi_2state(disps, pi, log_A, means, stds)

    changes = np.diff(states)
    starts = np.where(changes == -1)[0] + 1
    ends = np.where(changes == 1)[0] + 1

    if states[0] == 0:
        starts = np.insert(starts, 0, 0)
    if states[-1] == 0:
        ends = np.append(ends, n)

    out_rows = []
    for s, e in zip(starts, ends):
        seg = g.iloc[s:e]
        duration = (seg["time"].iloc[-1] - seg["time"].iloc[0]).total_seconds()
        if duration >= min_duration:
            out_rows.append({
                "agent_id": int(agent_id),
                "latitude": float(seg["latitude"].mean()),
                "longitude": float(seg["longitude"].mean()),
                "arrive_time": seg["time"].iloc[0],
                "leave_time": seg["time"].iloc[-1],
                "duration_s": float(duration),
                "n_points": int(len(seg)),
            })

    return out_rows


def get_stay_points(df, self_loop=0.95, mu_stop=15.0, sigma_stop=12.0,
                    mu_move=200.0, sigma_move=300.0, min_duration=300):
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

    pi = np.array([0.5, 0.5])
    A = np.array([[self_loop, 1 - self_loop],
                  [1 - self_loop, self_loop]])
    log_A = np.log(A)
    means = np.array([mu_stop, mu_move])
    stds = np.array([sigma_stop, sigma_move])

    groups = [(aid, g, pi, log_A, means, stds, min_duration)
              for aid, g in d.groupby("agent_id", sort=False)]
    print(f"Processing {len(groups)} agents with {cpu_count()} workers...")

    with Pool(cpu_count()) as pool:
        results = pool.map(_process_agent, groups)

    out_rows = [row for sub in results for row in sub]

    return pd.DataFrame(
        out_rows,
        columns=["agent_id", "latitude", "longitude", "arrive_time", "leave_time", "duration_s", "n_points"]
    )