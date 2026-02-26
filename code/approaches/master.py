

from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def get_stay_points(func=None, df=None, **kwargs):
    if df is None:
        raise ValueError("DataFrame is required")
    if func is None:
        raise ValueError("Function is required")
    
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


    groups = list(d.groupby("agent_id"))

    out_rows = []
    print(f"Processing {func.__name__} for {len(groups)} agents...")
    with ProcessPoolExecutor() as ex:
        for result in ex.map(func, groups, **kwargs):
            out_rows.extend(result)

    return pd.DataFrame(
        out_rows,
        columns=["agent_id", "latitude", "longitude", "arrive_time", "leave_time", "duration_s", "n_points"]
    )