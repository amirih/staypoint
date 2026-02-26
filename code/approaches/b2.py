
# Stay Point Detection method 1: 
# implementation of algorithm in 
# [1] Q. Li, Y. Zheng, X. Xie, Y. Chen, W. Liu, and W.-Y. Ma, "Mining user similarity based on location history", in Proceedings of the 16th ACM SIGSPATIAL international conference on Advances in geographic information systems, New York, NY, USA, 2008, pp. 34:1--34:10.

# Source: https://gist.github.com/RustingSword/5215046
# Adapted to fit the input data format and evaluation requirements of this project by Hossein Amiri

from math import radians, cos, sin, asin, sqrt

def getDistance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return 6371000 * c  # meters

def b2(agent_groups, dist_thresh_m=200, time_thresh_min=20, debug=False):
    agent_id, g = agent_groups
    time_thresh_s = float(time_thresh_min) * 60.0
    if debug:
        print(f"Processing agent_id: {agent_id} with {len(g)} records")
    found_points = []
    g = g.reset_index(drop=True)
    n = len(g)
    if n < 2:
        return found_points

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

            if dist > dist_thresh_m:
                deltaT = (g.at[j, "time"] - g.at[i, "time"]).total_seconds()

                if deltaT > time_thresh_s:
                    window = g.iloc[i:j+1]
                    found_points.append({
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
    if debug:
        print(f"Found {len(found_points)} stay points for agent_id: {agent_id}")
    return found_points



    