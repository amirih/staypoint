import math
# Copilot + ChatGPT-5.2 implementation
def haversine_m(lat1, lon1, lat2, lon2):
    # meters
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2)**2
    return 2 * R * math.asin(math.sqrt(a))


def b3(
    agent_groups,
    dist_thresh_m=200.0,     # spatial threshold (meters)
    time_thresh_min=20.0,    # temporal threshold (minutes)
    time_col="time",
    lat_col="latitude",
    lon_col="longitude",
    debug=False,
):
    """
    Classic stay point detection (Li et al.-style):
    - Start at i
    - Move j forward until distance(i, j) > dist_thresh_m
    - If time(i, j-1) >= time_thresh -> emit staypoint over i..j-1, set i = j
      else i += 1
    Expects g sorted by time.
    """
    agent_id, g = agent_groups
    if debug:
        print(f"Processing agent_id: {agent_id} with {len(g)} records")

    found_points = []
    g = g.sort_values(time_col).reset_index(drop=True)
    n = len(g)
    if n < 2:
        return found_points

    time_thresh_s = float(time_thresh_min) * 60.0
    i = 0

    while i < n - 1:
        j = i + 1

        while j < n:
            d = haversine_m(
                float(g.at[i, lat_col]), float(g.at[i, lon_col]),
                float(g.at[j, lat_col]), float(g.at[j, lon_col]),
            )

            if d > dist_thresh_m:
                # candidate window is i .. (j-1)
                deltaT = (g.at[j - 1, time_col] - g.at[i, time_col]).total_seconds()

                if deltaT >= time_thresh_s:
                    window = g.iloc[i:j]  # i..j-1
                    found_points.append({
                        "agent_id": int(agent_id),
                        "latitude": float(window[lat_col].mean()),
                        "longitude": float(window[lon_col].mean()),
                        "arrive_time": window[time_col].iloc[0],
                        "leave_time": window[time_col].iloc[-1],
                        "duration_s": float(deltaT),
                        "n_points": int(len(window)),
                        "radius_m": float(window.apply(
                            lambda r: haversine_m(
                                float(window[lat_col].mean()),
                                float(window[lon_col].mean()),
                                float(r[lat_col]),
                                float(r[lon_col]),
                            ),
                            axis=1
                        ).max()) if len(window) else 0.0,
                    })
                    i = j  # jump to first point outside radius
                else:
                    i += 1
                break

            j += 1

        # if we reached the end without breaking, no more staypoints starting at i
        if j >= n:
            break

    if debug:
        print(f"Found {len(found_points)} stay points for agent_id: {agent_id}")
    return found_points