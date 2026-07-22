import pandas as pd
import numpy as np
from pyproj import Transformer
import random

# Based on ChatGPT, need to verify. Default utm zone is for Atlanta
def apply_noise(df, avg_dist, utm_zone="EPSG:32616", lat_col='latitude', lon_col='longitude'):
    df = df.copy()

    # Need projection to flattened map and back to make dists easier to handle
    to_utm = Transformer.from_crs("EPSG:4326", utm_zone, always_xy=True)
    to_latlon = Transformer.from_crs(utm_zone, "EPSG:4326", always_xy=True)

    # These should be meters now? Or something, lol
    x, y = to_utm.transform(df[lon_col].values, df[lat_col].values)

    # In theory, this should give avg dist in Euclidean rather than each direction separately
    sigma = avg_dist / np.sqrt(np.pi / 2)

    # Now the actual noise
    dx = np.random.normal(0, sigma, size=len(df))
    dy = np.random.normal(0, sigma, size=len(df))
    x_noisy = x + dx
    y_noisy = y + dy

    lon_noisy, lat_noisy = to_latlon.transform(x_noisy, y_noisy)

    df['lat_noisy'] = lat_noisy
    df['lon_noisy'] = lon_noisy

    return df


def apply_dropouts(df, avg_dropouts_per_day=1, typical_dropout_time=1, num_days=30):
    n = len(df)
    avg_num_dropouts = avg_dropouts_per_day * num_days

    # Poisson distributed dropouts- say we want 30 dropouts (on average), than every point has probability 30/n of being selected
    k = np.random.poisson(avg_num_dropouts)
    starting_indices = sorted(random.sample(range(n), k))

    # Select dropout times from exponential distribution- roughly half will be shorter, and half longer than typical
    dropout_times = np.random.exponential(size=k)
    dropout_times = [pd.Timedelta(hours=(x*typical_dropout_time)) for x in dropout_times]

    df = df.copy()
    df = df.reset_index(drop=True)

    # Select the portion of the trajectory to drop for each dropout
    indices_to_drop=[]
    for ind, d_time in zip(starting_indices, dropout_times):
        start_time = df['time'].iloc[ind]
        to_drop = df[df['time'].between(start_time, start_time+d_time)]
        indices_to_drop.extend(to_drop.index)
    
    indices_to_drop=set(indices_to_drop)

    return df.drop(indices_to_drop, axis=0)

base_traj = pd.read_parquet('POL-ATL-data/noised_trajectory.parquet')
base_traj = base_traj.drop(columns=['n_lat', 'n_lon'])

# Noise will be, on average, "dist" meters from ground truth point
for dist in [10, 25, 50]:
    print(f'Running noising with avg distance of {dist}')
    noisy_traj = apply_noise(base_traj, 10)

    # First param is dropouts per day, second is typical dropout time (in hours)
    dropout_params = [(1, 1), (2, 2), (3, 4)]

    for i, params in enumerate(dropout_params):
        print(f'\tRunning params (level {i+1}): avg_dropouts_per_day={params[0]}, typical_dropout_time={params[1]} hours')

        all_reduced_trajs = []

        for name, group in noisy_traj.groupby('agent_id'):
            reduced_traj = apply_dropouts(group, avg_dropouts_per_day=params[0], typical_dropout_time=params[1])
            all_reduced_trajs.append(reduced_traj)
        
        all_reduced_trajs = pd.concat(all_reduced_trajs, ignore_index=True)
        all_reduced_trajs.to_parquet('POL-ATL-data/datasets/trajectories_noiselevel'+str(dist)+'_dropoutlevel'+str(i+1)+'.parquet')