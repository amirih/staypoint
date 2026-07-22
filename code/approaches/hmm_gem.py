import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


# ============================================================
# Distance helper
# ============================================================
def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(a))


# ============================================================
# Build arbitrary K-step displacement features
# ============================================================
def build_kstep_displacements(lats, lons, K):
    n = len(lats)
    obs = np.full((n, K), np.nan)
    for k in range(1, K + 1):
        if n > k:
            d = haversine_vec(lats[:-k], lons[:-k], lats[k:], lons[k:])
            obs[:-k, k - 1] = d
    return obs


# ============================================================
# Diagonal Gaussian log-likelihood
# ============================================================
def log_gaussian_diag(obs, means, stds):
    logp = np.zeros((obs.shape[0], 2))
    const = -np.log(stds * np.sqrt(2 * np.pi))
    for s in range(2):
        z = (obs - means[s]) / stds[s]
        per_dim = -0.5 * z**2 + const[s]
        per_dim = np.where(np.isnan(obs), 0.0, per_dim)
        logp[:, s] = per_dim.sum(axis=1)
    return logp


# ============================================================
# Forward–Backward (log space)
# ============================================================
def forward_backward(pi, A, log_emit):
    T = log_emit.shape[0]
    logA = np.log(A)

    alpha = np.zeros((T, 2))
    beta = np.zeros((T, 2))

    alpha[0] = np.log(pi) + log_emit[0]
    for t in range(1, T):
        for s in range(2):
            alpha[t, s] = log_emit[t, s] + np.logaddexp(
                alpha[t - 1, 0] + logA[0, s],
                alpha[t - 1, 1] + logA[1, s],
            )

    beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        for s in range(2):
            beta[t, s] = np.logaddexp(
                logA[s, 0] + log_emit[t + 1, 0] + beta[t + 1, 0],
                logA[s, 1] + log_emit[t + 1, 1] + beta[t + 1, 1],
            )

    log_gamma = alpha + beta
    log_gamma -= np.logaddexp(log_gamma[:, 0], log_gamma[:, 1])[:, None]
    return np.exp(log_gamma)


# ============================================================
# EM M-step for emissions (global)
# ============================================================
def mstep_emissions(obs, gamma, min_std=1.0):
    K = obs.shape[1]
    means = np.zeros((2, K))
    stds = np.zeros((2, K))

    for s in range(2):
        w = gamma[:, s]
        W = np.nansum(w[:, None] * (~np.isnan(obs)), axis=0)

        means[s] = np.nansum(w[:, None] * obs, axis=0) / W

        var = np.nansum(
            w[:, None] * (obs - means[s])**2,
            axis=0
        ) / W

        stds[s] = np.sqrt(np.maximum(var, min_std**2))

    return means, stds


# ============================================================
# Global EM training
# ============================================================
def train_global_em(all_obs, pi, A, means, stds, n_iter=5):
    for _ in range(n_iter):
        log_emit = log_gaussian_diag(all_obs, means, stds)
        gamma = forward_backward(pi, A, log_emit)

        # Learn only emissions (π and A frozen)
        means, stds = mstep_emissions(all_obs, gamma)

    return means, stds


# ============================================================
# Viterbi decoding
# ============================================================
def viterbi(obs, pi, A, means, stds):
    log_emit = log_gaussian_diag(obs, means, stds)
    logA = np.log(A)

    T = obs.shape[0]
    V = np.zeros((T, 2))
    B = np.zeros((T, 2), dtype=int)

    V[0] = np.log(pi) + log_emit[0]

    for t in range(1, T):
        for s in range(2):
            scores = V[t - 1] + logA[:, s]
            B[t, s] = np.argmax(scores)
            V[t, s] = scores[B[t, s]] + log_emit[t, s]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(V[-1])
    for t in range(T - 2, -1, -1):
        states[t] = B[t + 1, states[t + 1]]

    return states


# ============================================================
# Per-agent segmentation
# ============================================================
def _process_agent(args):
    agent_id, g, pi, A, means, stds, K, min_duration = args
    g = g.reset_index(drop=True)

    obs = build_kstep_displacements(
        g["latitude"].values,
        g["longitude"].values,
        K
    )

    states = viterbi(obs, pi, A, means, stds)

    changes = np.diff(states)
    starts = np.where(changes == -1)[0] + 1
    ends = np.where(changes == 1)[0] + 1

    if states[0] == 0:
        starts = np.insert(starts, 0, 0)
    if states[-1] == 0:
        ends = np.append(ends, len(g))

    rows = []
    for s, e in zip(starts, ends):
        seg = g.iloc[s:e]
        duration = (seg["time"].iloc[-1] - seg["time"].iloc[0]).total_seconds()
        if duration >= min_duration:
            rows.append({
                "agent_id": agent_id,
                "latitude": seg["latitude"].mean(),
                "longitude": seg["longitude"].mean(),
                "arrive_time": seg["time"].iloc[0],
                "leave_time": seg["time"].iloc[-1],
                "duration_s": duration,
                "n_points": len(seg),
            })

    return rows


# ============================================================
# Public API
# ============================================================
def hmm_gem(
    df,
    K=3,
    min_duration=300,
    self_loop=0.97,
    em_iters=5,
    **kargs
):
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"])
    d = d.drop(columns=["latitude", "longitude"], errors="ignore")
    d = d.rename(columns={"n_lat": "latitude", "n_lon": "longitude"})
    d = d.sort_values(["agent_id", "time"]).reset_index(drop=True)

    pi = np.array([0.5, 0.5])
    A = np.array([
        [self_loop, 1 - self_loop],
        [1 - self_loop, self_loop],
    ])

    # ---- Initial emissions (domain-informed)
    means = np.vstack([
        np.linspace(15, 15 + 10*(K-1), K),
        np.linspace(200, 200 + 150*(K-1), K),
    ])
    stds = np.vstack([
        np.linspace(12, 12 + 15*(K-1), K),
        np.linspace(300, 300 + 200*(K-1), K),
    ])

    # ---- GLOBAL EM (single pass)
    all_obs = []
    for _, g in d.groupby("agent_id"):
        obs = build_kstep_displacements(
            g["latitude"].values,
            g["longitude"].values,
            K
        )
        all_obs.append(obs)

    all_obs = np.vstack(all_obs)

    means, stds = train_global_em(
        all_obs, pi, A, means, stds, n_iter=em_iters
    )

    # ---- Per-agent decoding (parallel)
    groups = [
        (aid, g, pi, A, means, stds, K, min_duration)
        for aid, g in d.groupby("agent_id", sort=False)
    ]

    with Pool(cpu_count()) as pool:
        results = pool.map(_process_agent, groups)

    return pd.DataFrame([r for sub in results for r in sub])
