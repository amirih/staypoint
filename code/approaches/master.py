

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd

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