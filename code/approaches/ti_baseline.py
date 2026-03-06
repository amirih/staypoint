# Imports
import pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd

from trackintel import Positionfixes, Staypoints
from trackintel.preprocessing import generate_locations, generate_staypoints, generate_triplegs, merge_staypoints

def ti_POL_ATL_data(traj):
    traj = traj.sort_values(by=['time']).reset_index(drop=True)

    traj.drop(columns=['tick', 'latitude', 'longitude'], inplace=True)
    traj.rename(columns={'time':'tracked_at', 'agent_id':'user_id'}, inplace=True)
    traj['tracked_at'] = traj['tracked_at'].dt.tz_localize('US/Eastern')
    traj = gpd.GeoDataFrame(traj, geometry=gpd.points_from_xy(traj['n_lon'], traj['n_lat'], crs="EPSG:4326"))

    # Will return pfs as the original df but basically with assigned staypoint id (or NA) to each point
    # Generates initial sps
    pfs, sps = generate_staypoints(
        traj,
        dist_threshold=50, # Min dist between staypoints, in meters
        time_threshold=pd.Timedelta(minutes=5), # Min duration to create a staypoint
        gap_threshold=pd.Timedelta(minutes=25), # Max gap time to still mark something as a staypoint
        include_last=True, # Makes sure we include the last one if the user ends there
    )

    # If no staypoints detected, nothing to do
    if len(sps) == 0:
        return None

    # Adds a location id to each sp so we can decide if they're the same place and consecutive to make merging easier
    # Taking default parameters for now
    sps, _ = generate_locations(sps)
    
    # Generates triplegs (which we'll need later) by just looking between sps
    # Returns pfs as initial pfs but with assigned tripleg id (or NA) to each point. Now, every point should be at a sp or tripleg
    pfs, tpls = generate_triplegs(pfs, gap_threshold=25)

    # Merges staypoints that are at the same location, consecutive, and within some time gap of each other
    sps = merge_staypoints(sps, tpls, max_time_gap=pd.Timedelta(minutes=25), agg={"geometry":"last"})
    
    return sps

# POL-ATL-data
trajectories = pd.read_parquet('POL-ATL-data/noised_trajectory.parquet')

all_sps = []

for name, group in trajectories.groupby('agent_id'):
    agent_sps = ti_POL_ATL_data(group)
    if agent_sps is not None:
        all_sps.append(agent_sps)

all_sps = pd.concat(all_sps, ignore_index=True)
all_sps.to_csv('POL-ATL-data/attempts/ti_50m_5min_25g.csv')

def ti_sim1_data(traj):
    traj = traj.sort_values(by=['timestamp']).reset_index(drop=True)

    traj.rename(columns={'timestamp':'tracked_at', 'agent':'user_id'}, inplace=True)
    traj = gpd.GeoDataFrame(traj, geometry=gpd.points_from_xy(traj['longitude'], traj['latitude'], crs="EPSG:4326"))

    # These parameters are supposeed to be optimized to this data
    pfs, sps = generate_staypoints(
        traj,
        dist_threshold=50, # Min dist between staypoints, in meters
        time_threshold=pd.Timedelta(minutes=5), # Min duration to create a staypoint
        gap_threshold=pd.Timedelta(minutes=25), # Max gap time to still mark something as a staypoint
        include_last=True, # Makes sure we include the last one if the user ends there
    )

    if len(sps) == 0:
        return None

    sps, _ = generate_locations(sps)
    
    pfs, tpls = generate_triplegs(pfs, gap_threshold=25)

    sps = merge_staypoints(sps, tpls, max_time_gap=pd.Timedelta(minutes=25), agg={"geometry":"last"})
    
    return sps

# sim1 data
reader = pq.ParquetFile('sim1/data.zstd.parquet')
batch_size = 6048000

all_sps = []

prev_df = None
for batch in reader.iter_batches(batch_size=batch_size):
    df = batch.to_pandas()

    df = pd.concat([prev_df, df], ignore_index=True)

    prev_df = df[df.agent == df.agent.iloc[-1]]
    df = df[df.agent != df.agent.iloc[-1]].reset_index(drop=True)

    for name, group in df.groupby('agent'):
        agent_sps = ti_sim1_data(group)
        if agent_sps is not None:
            all_sps.append(agent_sps)

# Add last agent
agent_sps = ti_sim1_data(prev_df)
if agent_sps is not None:
    all_sps.append(agent_sps)

all_sps = pd.concat(all_sps, ignore_index=True)
all_sps.to_csv('sim1/attempts/ti_50m_5min_25g.csv')