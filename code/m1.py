import pandas as pd
from math import radians, cos, sin, asin, sqrt
from utils import print_time as print

def getDistance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c  # meters

def get_stay_points(df, distThres=200, timeThres=20*60):
    required = {"agent_id", "latitude", "longitude", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = df.copy()

    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time", "latitude", "longitude", "agent_id"])
    d = d.sort_values(["agent_id", "time"]).reset_index(drop=True)

    out_rows = []

    for agent_id, g in d.groupby("agent_id", sort=False):
        print(f"Processing agent_id: {agent_id} with {len(g)} records")
        found_points = len(out_rows)
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue

        i = 0

        while i < n - 1:
            j = i + 1
            found = False

            while j < n:
                # IMPORTANT: getDistance expects (lon, lat, lon, lat)
                dist = getDistance(
                    g.at[i, "longitude"], g.at[i, "latitude"],
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
                        i = j  # jump
                        found = True
                    break

                j += 1

            i = i + 1 if not found else i + 1  # match original behavior (+1 after jump)

        print(f"Found {len(out_rows) - found_points} stay points for agent_id: {agent_id}")

    return pd.DataFrame(
        out_rows,
        columns=["agent_id", "latitude", "longitude", "arrive_time", "leave_time", "duration_s", "n_points"]
    )
