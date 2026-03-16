import math
import numpy as np


# Copilot + ChatGPT-5.2 implementation
def haversine_m(lat1, lon1, lat2, lon2):
    # meters
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def b3_adaptive(
    agent_groups,
    alpha=3.0,                 # scaling factor for adaptive radius
    dist_thresh_m=50.0,                # minimum allowed radius (meters)
    time_thresh_min=20.0,
    noise_window=10,           # rolling window size for std estimation
    time_col="time",
    lat_col="latitude",
    lon_col="longitude",
    debug=False,
):
    """
    Adaptive Radius Stay Detection:

    r_i = max(r_min, alpha * local_spatial_std(i))

    Uses two-pointer scanning similar to b3,
    but spatial threshold varies per starting point.
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

    # ---- Precompute projected coordinates (meters) ----
    mean_lat = g[lat_col].mean()
    lat_to_m = 110540.0
    lon_to_m = 111320.0 * math.cos(math.radians(mean_lat))

    x = g[lon_col].values * lon_to_m
    y = g[lat_col].values * lat_to_m

    coords_m = np.column_stack([x, y])

    # ---- Compute local rolling std ----
    local_std = np.zeros(n)

    for i in range(n):
        left = max(0, i - noise_window)
        right = min(n, i + noise_window + 1)
        window = coords_m[left:right]

        if len(window) > 1:
            std_x = np.std(window[:, 0])
            std_y = np.std(window[:, 1])
            local_std[i] = np.sqrt(std_x**2 + std_y**2)
        else:
            local_std[i] = 0.0

    # ---- Main adaptive scanning ----
    i = 0
    while i < n - 1:

        # Adaptive radius for this anchor
        r_min = dist_thresh_m
        r_i = max(r_min, alpha * local_std[i])

        j = i + 1

        while j < n:
            d = haversine_m(
                float(g.at[i, lat_col]), float(g.at[i, lon_col]),
                float(g.at[j, lat_col]), float(g.at[j, lon_col]),
            )
            import ipdb; ipdb.set_trace()
            if d > r_i:
                deltaT = (
                    g.at[j - 1, time_col]
                    - g.at[i, time_col]
                ).total_seconds()

                if deltaT >= time_thresh_s:
                    window = g.iloc[i:j]

                    centroid_lat = float(window[lat_col].mean())
                    centroid_lon = float(window[lon_col].mean())

                    found_points.append({
                        "agent_id": int(agent_id),
                        "latitude": centroid_lat,
                        "longitude": centroid_lon,
                        "arrive_time": window[time_col].iloc[0],
                        "leave_time": window[time_col].iloc[-1],
                        "duration_s": float(deltaT),
                        "n_points": int(len(window)),
                        "adaptive_radius_m": float(r_i),
                    })

                    i = j
                else:
                    i += 1
                break

            j += 1

        if j >= n:
            break

    if debug:
        print(f"Found {len(found_points)} stay points for agent_id: {agent_id}")

    return found_points


def b3(
    agent_groups,
    dist_thresh_m=200.0,     # spatial threshold (meters)
    time_thresh_min=20.0,    # temporal threshold (minutes)
    time_col="time",
    lat_col="latitude",
    lon_col="longitude",
    debug=True,
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
            import ipdb; ipdb.set_trace()
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