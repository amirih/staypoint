# Imports
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn import metrics
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

def min_max_normalize(col, new_min=0, new_max=1):
    old_min = col.min()
    old_max = col.max()
    return (((col - old_min) / (old_max - old_min)) * (new_max - new_min)) + new_min

def run_clustering_pipeline(traj, sps, eps=0.0007, min_sp_time=5, avg_freq=60, time_col='time', lat='latitude', lon='longitude'):
    # Find approximately how many points we want as minimum for cluster
    # Will be harder with real-world data and inconsistent frequencies
    min_sp_time = pd.Timedelta(minutes=min_sp_time)
    min_samples = int(min_sp_time.total_seconds() / avg_freq)

    # For sanity
    traj = traj.sort_values(by=time_col).reset_index(drop=True)

    # Add GT sps IDs to traj
    if sps is not None:
        traj['cluster_id'] = -1
        for idx, row in sps.iterrows():
            traj.loc[(traj[time_col] >= row.startTime) & (traj[time_col] <= row.endTime), 'cluster_id'] = idx
    
    # Make time numerical
    sim_start = traj[time_col].min()
    traj['norm_time'] = traj[time_col].apply(lambda x: (x-sim_start).total_seconds())

    # Normalize columns to cluster
    traj['norm_time'] = min_max_normalize(traj.norm_time)
    traj['norm_lat'] = min_max_normalize(traj[lat])
    traj['norm_lon'] = min_max_normalize(traj[lon])

    # CLustering
    to_cluster = traj[['norm_time', 'norm_lat', 'norm_lon']]

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(to_cluster)
    #clustering = OPTICS(min_samples=min_samples).fit(to_cluster)
    labels = clustering.labels_

    traj['prediction'] = labels
    next_cluster = max(labels)+1
    updates = {}

    # Clusters are not contiguous in terms of time, so we ammend that
    for predicted_cluster, cluster in traj.groupby('prediction'):
        if predicted_cluster == -1: # Non-staypoint/noise
            continue

        cluster = cluster.reset_index() # Now we have a column called 'index' representing the index in the original df
        cluster['gap'] = cluster['index'].diff() > 1
        
        # To check time gaps
        cluster['t_diff'] = cluster[time_col].diff()

        # We may have to separate this cluster into multiple clusters
        cluster['sub_cluster'] = predicted_cluster

        # Check each gap
        problems = cluster[cluster.gap]
        for idx, row in problems.iterrows():
            # If gap is large, we want to separate clusters
            if row['t_diff'] > min_sp_time:
                cluster.loc[cluster['index'] >= row['index'], 'sub_cluster'] = next_cluster
                next_cluster += 1

        for sub_cluster_id, sub_cluster in cluster.groupby('sub_cluster'):
            # If too short, convert to noise
            if sub_cluster[time_col].max() - sub_cluster[time_col].min() < min_sp_time:
                updates.setdefault(-1, []).append((sub_cluster['index'].min(), sub_cluster['index'].max()))
            # Otherwise, we want to cover all the gaps in the cluster
            else:
                #updates.setdefault(sub_cluster_id, []).append((sub_cluster['index'].min(), sub_cluster['index'].max()))
                updates[sub_cluster_id] = (sub_cluster['index'].min(), sub_cluster['index'].max())

    # First, convert noise
    if -1 in updates.keys():
        for tup in updates[-1]:
            traj.loc[tup[0]:tup[1], 'prediction'] = -1
        del updates[-1]

    # Second, convert sub clusters and mark all missed points to make clusters contiguous
    for cluster, update in updates.items():
        traj.loc[update[0]:update[1], 'prediction'] = cluster

    return traj


# ===== POL ATL DATA =====
noised_trajs = pd.read_parquet('POL-ATL-data/noised_trajectory.parquet')
all_sps = pd.read_parquet('POL-ATL-data/naive_sps.parquet')

pred_sps = pd.DataFrame(columns=['agent_id',
                                 'startTime',
                                 'endTime',
                                 'mean_lat',
                                 'mean_lon',
                                 'med_lat',
                                 'med_lon'])

for agent_id in noised_trajs.agent_id.unique():
    sample_traj = noised_trajs[noised_trajs.agent_id == agent_id]
    sample_sps = all_sps[all_sps.agent_id == agent_id]
    sample_traj = run_clustering_pipeline(sample_traj, sample_sps, eps=0.005, time_col='time', lat='n_lat', lon='n_lon')

    # ===== Fill predicted SPS dataframe =====
    for pred, sp in sample_traj.groupby('prediction'):
        if pred == -1:
            continue
        new_sp = [agent_id, sp.time.min(), sp.time.max(), sp.n_lat.mean(), sp.n_lon.mean(), sp.n_lat.median(), sp.n_lon.median()]
        pred_sps.loc[len(pred_sps)] = new_sp

pred_sps.to_parquet('POL-ATL-data/attempts/temporal-dbscan.parquet')

# ===== SIM 1 DATA =====
reader = pq.ParquetFile('sim1/data.zstd.parquet')
batch_size = 6048000

pred_sps = pd.DataFrame(columns=['agent_id',
                                 'startTime',
                                 'endTime',
                                 'mean_lat',
                                 'mean_lon',
                                 'med_lat',
                                 'med_lon'])

prev_df = None # Handling borders

for batch in reader.iter_batches(batch_size=batch_size):
    df = batch.to_pandas()

    df = pd.concat([prev_df, df], ignore_index=True)

    prev_df = df[df.agent == df.agent.iloc[-1]]
    df = df[df.agent != df.agent.iloc[-1]].reset_index(drop=True)

    for name, group in df.groupby('agent'):
        agent_sps = run_clustering_pipeline(group, None, eps=0.005, avg_freq=10, time_col='timestamp')
        if agent_sps is not None:
            # ===== Fill predicted SPS dataframe =====
            for pred, sp in agent_sps.groupby('prediction'):
                if pred == -1:
                    continue
                new_sp = [name, sp.timestamp.min(), sp.timestamp.max(), sp.latitude.mean(), sp.longitude.mean(), sp.latitude.median(), sp.longitude.median()]
                pred_sps.loc[len(pred_sps)] = new_sp

# Add last agent
agent_sps = run_clustering_pipeline(prev_df, None, eps=0.005, time_col='timestamp')
if agent_sps is not None:
    # ===== Fill predicted SPS dataframe =====
    for pred, sp in agent_sps.groupby('prediction'):
        if pred == -1:
            continue
        new_sp = [sp.agent.iloc[0], sp.timestamp.min(), sp.timestamp.max(), sp.latitude.mean(), sp.longitude.mean(), sp.latitude.median(), sp.longitude.median()]
        pred_sps.loc[len(pred_sps)] = new_sp

pred_sps.to_parquet('sim1/temporal-dbscan.parquet')