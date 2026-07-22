# Imports
import pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd

from trackintel import Positionfixes, Staypoints
from trackintel.preprocessing import generate_locations, generate_staypoints, generate_triplegs, merge_staypoints

def ti_POL_ATL_data(traj, time_col='time', lon_col='n_lon', lat_col='n_lat'):
    traj = traj.sort_values(by=[time_col]).reset_index(drop=True)

    if lat_col != 'latitude' and 'latitude' in traj.columns:
        traj.drop(columns=['tick', 'latitude', 'longitude'], inplace=True)
    else:
        traj.drop(columns=['tick'], inplace=True)
    traj.rename(columns={time_col:'tracked_at', 'agent_id':'user_id'}, inplace=True)
    traj['tracked_at'] = traj['tracked_at'].dt.tz_localize('US/Eastern')
    traj = gpd.GeoDataFrame(traj, geometry=gpd.points_from_xy(traj[lon_col], traj[lat_col], crs="EPSG:4326"))

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
    # Gives an error if insufficient number of staypoints or no time between them
    # Returns pfs as initial pfs but with assigned tripleg id (or NA) to each point. Now, every point should be at a sp or tripleg
    try:
        pfs, tpls = generate_triplegs(pfs, gap_threshold=25)
    except:
        return sps

    # Merges staypoints that are at the same location, consecutive, and within some time gap of each other
    sps = merge_staypoints(sps, tpls, max_time_gap=pd.Timedelta(minutes=25), agg={"geometry":"last"})
    
    return sps


if __name__ == "__main__":
    # POL-ATL-data
    trajectories = pd.read_parquet('POL-ATL-data/noised_trajectory.parquet')

    all_sps = []

    for name, group in trajectories.groupby('agent_id'):
        agent_sps = ti_POL_ATL_data(group)
        if agent_sps is not None:
            all_sps.append(agent_sps)

    all_sps = pd.concat(all_sps, ignore_index=True)
    all_sps.to_csv('POL-ATL-data/attempts/ti_50m_5min_25g.csv')
    