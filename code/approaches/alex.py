import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

class StayPointDetector:
    def __init__(self, eps_meters=40, min_time_minutes=5, max_gap_minutes=5):
        """
        Args:
            eps_meters (float): DBSCAN epsilon. 40-60m is recommended for GPS noise.
            min_time_minutes (int): Minimum duration to be considered a staypoint.
            max_gap_minutes (int): Max allowed gap within a cluster before splitting.
        """
        self.eps_meters = eps_meters
        self.min_time_minutes = min_time_minutes
        self.max_gap_minutes = max_gap_minutes
        self.EARTH_RADIUS = 6371000

    def fit_predict(self, df):

        process_df = df[['time', 'agent_id', 'n_lat', 'n_lon']].copy()
        
        if not pd.api.types.is_datetime64_any_dtype(process_df['time']):
            process_df['time'] = pd.to_datetime(process_df['time'])
            
        process_df = process_df.sort_values(['agent_id', 'time']).reset_index(drop=True)
        
        results = []
        
        for agent_id, agent_data in tqdm(process_df.groupby('agent_id'), desc="Detecting Staypoints"):
            if len(agent_data) < 2:
                continue

            coords = np.radians(agent_data[['n_lat', 'n_lon']].values)
            eps_rad = self.eps_meters / self.EARTH_RADIUS
            
            db = DBSCAN(eps=eps_rad, min_samples=5, metric='haversine', algorithm='ball_tree')
            labels = db.fit_predict(coords)
            
            agent_data = agent_data.copy()
            agent_data['cluster'] = labels
            
            valid_clusters = agent_data[agent_data['cluster'] != -1]
            if valid_clusters.empty:
                continue

            for cluster_id, cluster_pts in valid_clusters.groupby('cluster'):
                cluster_pts = cluster_pts.sort_values('time')
                
                time_diffs = cluster_pts['time'].diff().dt.total_seconds() / 60.0
                session_ids = (time_diffs > self.max_gap_minutes).cumsum().fillna(0)
                
                for _, session in cluster_pts.groupby(session_ids):
                    start_t = session['time'].iloc[0]
                    end_t = session['time'].iloc[-1]
                    duration = (end_t - start_t).total_seconds() / 60.0
                    
                    if duration >= self.min_time_minutes:
                        est_lat = session['n_lat'].mean()
                        est_lon = session['n_lon'].mean()
                        
                        results.append({
                            'agent_id': agent_id,
                            'pred_start': start_t,
                            'pred_end': end_t,
                            'pred_lat': est_lat,
                            'pred_lon': est_lon,
                            'duration': duration
                        })
                        
        return pd.DataFrame(results)

# ------------------------------------------------------------------------------
# Evaluation 
# ------------------------------------------------------------------------------
def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in meters"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def evaluate_performance(pred_df, truth_df):
    """
    Compares detected staypoints against naive_sps.parquet (Ground Truth).
    Matches are determined by Temporal Overlap.
    """
    print("\nStarting Evaluation...")
    

    p_df = pred_df.copy()
    t_df = truth_df.copy()
    
    if 'pred_start' in p_df.columns: 
        p_df['start'] = pd.to_datetime(p_df['pred_start'])
        p_df['end'] = pd.to_datetime(p_df['pred_end'])
    elif 'startTime' in p_df.columns:
        p_df['start'] = pd.to_datetime(p_df['startTime'])
        p_df['end'] = pd.to_datetime(p_df['endTime'])
        
    if 'startTime' in t_df.columns: 
        t_df['start'] = pd.to_datetime(t_df['startTime'])
        t_df['end'] = pd.to_datetime(t_df['endTime'])
    
    tp = 0  
    fp = 0  
    fn = 0  
    spatial_errors = []
    
    agents = set(p_df['agent_id']).union(set(t_df['agent_id']))
    
    for agent in tqdm(agents, desc="Evaluating"):
        p_agent = p_df[p_df['agent_id'] == agent]
        t_agent = t_df[t_df['agent_id'] == agent]
        
        matched_gt_indices = set()
        
        for _, p_row in p_agent.iterrows():
            has_match = False
            
            for t_idx, t_row in t_agent.iterrows():
                latest_start = max(p_row['start'], t_row['start'])
                earliest_end = min(p_row['end'], t_row['end'])
                
                if latest_start < earliest_end:
                    has_match = True
                    matched_gt_indices.add(t_idx)
                    

                    dist = haversine_np(p_row['pred_lat'], p_row['pred_lon'], 
                                        t_row['latitude'], t_row['longitude'])
                    spatial_errors.append(dist)

            
            if has_match:
                tp += 1
            else:
                fp += 1
        
        fn += len(t_agent) - len(matched_gt_indices)
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_spatial_error = np.mean(spatial_errors) if spatial_errors else 0.0
    
    print("-" * 40)
    print(f"Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Avg Spatial Error: {avg_spatial_error:.2f} meters")
    print("-" * 40)
    
    return {
        'precision': precision, 
        'recall': recall, 
        'f1': f1, 
        'spatial_error': avg_spatial_error
    }


if __name__ == "__main__":
    
        traj_df = pd.read_parquet('POL-ATL-data/noised_trajectory.parquet')
        sps_truth_df = pd.read_parquet('POL-ATL-data/naive_sps.parquet')
        
        
        
        sample_agents = traj_df['agent_id'].unique()[:500]
        traj_df = traj_df[traj_df['agent_id'].isin(sample_agents)]
        sps_truth_df = sps_truth_df[sps_truth_df['agent_id'].isin(sample_agents)]



        print(f"Loaded {len(traj_df)} trajectory points and {len(sps_truth_df)} ground truth staypoints.")


        detector = StayPointDetector(eps_meters=30, min_time_minutes=5)
        
        detected_sps = detector.fit_predict(traj_df)
        
        if detected_sps.empty:
            print("No staypoints detected. Try increasing eps_meters.")
        else:
            print(f"Detected {len(detected_sps)} staypoints.")
            
            metrics = evaluate_performance(detected_sps, sps_truth_df)
            

            print("\nValidation on Trajectory Unnoised Columns (Sample):")
            sample_agent = detected_sps.iloc[0]['agent_id']
            sample_start = detected_sps.iloc[0]['pred_start']
            sample_end = detected_sps.iloc[0]['pred_end']
            
            mask = (traj_df['agent_id'] == sample_agent) & \
                   (traj_df['time'] >= sample_start) & \
                   (traj_df['time'] <= sample_end)
            
            true_traj_segment = traj_df[mask]
            true_mean_lat = true_traj_segment['latitude'].mean()
            true_mean_lon = true_traj_segment['longitude'].mean()
            
            pred_lat = detected_sps.iloc[0]['pred_lat']
            pred_lon = detected_sps.iloc[0]['pred_lon']
            
            err = haversine_np(pred_lat, pred_lon, true_mean_lat, true_mean_lon)
            print(f"Sample Interval Error (Pred vs Actual Traj Mean): {err:.2f} meters")
            
            output_df = detected_sps.rename(columns={
                'pred_start': 'startTime',  
                'pred_end': 'endTime',
                'pred_lat': 'latitude',
                'pred_lon': 'longitude'})

            output_df['startTime'] = pd.to_datetime(output_df['startTime'])
            output_df['endTime'] = pd.to_datetime(output_df['endTime'])

            output_filename = "POL-ATL-data/detected_staypoints.parquet"
            output_df.to_parquet(output_filename, index=False)





