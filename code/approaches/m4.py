# Stay Point Detection method 4:
# Centroid-based sliding window
# Instead of measuring distance from the anchor point to the new point,
# measure from the running centroid of the window to the new point.

import pandas as pd
from math import radians, cos, sin, asin, sqrt
from multiprocessing import Pool, cpu_count
from utils import print_time as print


def getDistance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c


def _process_agent(args):
    agent_id, g, distThres, timeThres = args
    g = g.reset_index(drop=True)
    n = len(g)
    if n < 2:
        return []

    out_rows = []
    i = 0
    while i < n - 1:
        j = i + 1
        found = False
        sum_lat = g.at[i, "latitude"]
        sum_lon = g.at[i, "longitude"]
        count = 1

        while j < n:
            centroid_lat = sum_lat / count
            centroid_lon = sum_lon / count

            dist = getDistance(
                centroid_lon, centroid_lat,
                g.at[j, "longitude"], g.at[j, "latitude"]
            )

            if dist > distThres:
                deltaT = (g.at[j, "time"] - g.at[i, "time"]).total_seconds()
                if deltaT > timeThres:
                    window = g.iloc[i:j+1]
                    out_rows.append({
                        "agent_id": int(agent_id),
                        "latitude": float(window["latitude"].mean()),
                        "longitude": float(window["longitude"].mean()),
                        "arrive_time": window["time"].iloc[0],
                        "leave_time": window["time"].iloc[-1],
                        "duration_s": float(deltaT),
                        "n_points": int(len(window)),
                    })
                    i = j
                    found = True
                break

            sum_lat += g.at[j, "latitude"]
            sum_lon += g.at[j, "longitude"]
            count += 1
            j += 1

        i = i + 1 if not found else i + 1

    return out_rows


def get_stay_points(df, distThres=200, timeThres=20*60):
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

    groups = [(aid, g, distThres, timeThres) for aid, g in d.groupby("agent_id", sort=False)]
    print(f"Processing {len(groups)} agents with {cpu_count()} workers...")

    with Pool(cpu_count()) as pool:
        results = pool.map(_process_agent, groups)

    out_rows = [row for sub in results for row in sub]

    return pd.DataFrame(
        out_rows,
        columns=["agent_id", "latitude", "longitude", "arrive_time", "leave_time", "duration_s", "n_points"]
    )
