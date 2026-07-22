import pandas as pd

def smooth_single_trajectory(traj, time_col='time', lat_col='latitude', lon_col='longitude', period='5min'):
    traj = traj.sort_values(by=time_col).reset_index(drop=True)

    traj['smooth_lat_mean'] = traj.rolling(period, on=time_col, min_periods=1)[lat_col].mean()
    traj['smooth_lon_mean'] = traj.rolling(period, on=time_col, min_periods=1)[lon_col].mean()

    traj['smooth_lat_median'] = traj.rolling(period, on=time_col, min_periods=1)[lat_col].median()
    traj['smooth_lon_median'] = traj.rolling(period, on=time_col, min_periods=1)[lon_col].median()

    return traj

def smooth_trajectories(trajs, agent_col='agent_id', time_col='time', lat_col='latitude', lon_col='longitude', period='5min'):
    smoothed_trajs = []

    for name, group in trajs.groupby(agent_col):
        group = smooth_single_trajectory(group, time_col=time_col, lat_col=lat_col, lon_col=lon_col, period=period)
        smoothed_trajs.append(group)
    
    smoothed_trajs = pd.concat(smoothed_trajs, ignore_index=True)
    return smoothed_trajs

if __name__ == '__main__':
    POL_trajs = pd.read_parquet('POL-ATL-data/noised_trajectory.parquet')
    smoothed_trajs = smooth_trajectories(POL_trajs, lat_col='n_lat', lon_col='n_lon')
