import math
import numpy as np


from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd


def grow(std,g=10):
    res = std**2
    for p in range(g, 1 - 1, -1):
        s = sum(std**k for k in range(1, p + 1))
        res += s ** (1 / p)

    for i in range(2, g + 1):
         res += (res/i)**(1/2)
    res/=(g**2)
    return round(res)
    

def get_best_distance_threshold(df, lat_col, lon_col):
    coords = df[[lat_col, lon_col]].to_numpy()
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])
    lat_diff = np.diff(lat_rad)
    lon_diff = np.diff(lon_rad)
    a = np.sin(lat_diff / 2) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(lon_diff / 2) ** 2
    distances = 2 * 6371000 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    std = distances.std()
    threshold = grow(std)
    return threshold + 50


def get_best_time_threshold(g, lat_col, lon_col, time_col):
    g = g.sort_values(time_col)
    g = g.drop_duplicates(subset=[lat_col, lon_col, time_col])
    time_diff = g[time_col].diff().dt.total_seconds().dropna()
    std = time_diff.std() 
    threshold = grow(std* get_best_distance_threshold(g, lat_col, lon_col))
    threshold = round(((threshold - (threshold/3)) / 60)**(1/4))
    return threshold



def get_stay_points(func=None, df=None, **kwargs):
    if df is None:
        raise ValueError("DataFrame is required")
    if func is None:
        raise ValueError("Function is required")
    
    required = {"agent_id", "latitude", "longitude", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = df.copy()

    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time", "latitude", "longitude", "agent_id"])
    d = d.sort_values(["agent_id", "time"]).reset_index(drop=True)

    time_thresh_min = kwargs.get("time_thresh_min")
    dist_thresh_m = kwargs.get("dist_thresh_m")
    if time_thresh_min is None:
        time_thresh_min = get_best_time_threshold(d, "latitude", "longitude", "time")
        kwargs["time_thresh_min"] = time_thresh_min
        print(f"Estimated time threshold: {time_thresh_min} minutes")

    if dist_thresh_m is None:
        dist_thresh_m = get_best_distance_threshold(d, "latitude", "longitude")
        kwargs["dist_thresh_m"] = dist_thresh_m
        print(f"Estimated distance threshold: {dist_thresh_m} meters")

    groups = list(d.groupby("agent_id"))

    out_rows = []
    print(f"Processing {func.__name__} for {len(groups)} agents...")
    func_with_params = partial(func, **kwargs) 
    with ProcessPoolExecutor() as ex:
        for result in ex.map(func_with_params, groups):
            out_rows.extend(result)

    return pd.DataFrame(
        out_rows,
        columns=["agent_id", "latitude", "longitude", "arrive_time", "leave_time", "duration_s", "n_points"]
    )