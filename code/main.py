import pyarrow.parquet as parquet
import pandas
import approaches.master as master
from approaches.original_sw import sw 
from approaches.hossein import hsw
import eval as eval_utils
import os
from utils import print_time as print
import utils 
utils.install()

def get_df(data_path = "data/v1/trajectory.parquet"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the file exists.")
    if data_path.endswith('.parquet'):
        df = parquet.read_table(data_path).to_pandas()
    elif data_path.endswith('.csv'):
        df = pandas.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .parquet")
    return df

def get_processed_trajectory_df(df):
    required = {"agent_id", "latitude", "longitude", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["time"] = pandas.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude", "agent_id"])
    df = df.sort_values(["agent_id", "time"]).reset_index(drop=True)

    return df

def save_df(df, output_path = "data/v1/b2/sp1.csv"):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .parquet")
        
def calculate_stay_points(func=None, input_df=None, output_path = "data/v1/b2/sp2.csv", **kwargs):
    if os.path.exists(output_path):
        print(f"Output already exists at {output_path}. Skipping calculation.")
        return
    sdf = master.get_stay_points(func=func, df=input_df, **kwargs)
    save_df(sdf, output_path=output_path)

def evaluate(calculated_data_path = "data/v1/b2/sp2.csv",ground_truth_path="data/v1/ground_truth.parquet"):
    output_dir = os.path.dirname(calculated_data_path)
    file_name = os.path.basename(calculated_data_path)
    json_path = f"{file_name}_score.json"
    if os.path.exists(os.path.join(output_dir, json_path)):
        print(f"Evaluation score already exists at {os.path.join(output_dir, json_path)}. Skipping evaluation.")
        return
    gt_df = get_df(ground_truth_path)
    gt_df.rename(columns={'startTime': 'arrive_time', 'endTime': 'leave_time'}, inplace=True)
    calc_df = get_df(calculated_data_path)
    print("Calculating evaluation score...")
    score = eval_utils.get_score(gt_df, calc_df)
    utils.save_json(score, os.path.join(output_dir, json_path))
    print(f"Evaluation Score: {score}")

if __name__ == "__main__":
    funcs=[hsw]
    time_thresholds = [100, 50, 25, 10, 5]
    distance_thresholds = [200, 100, 50]
    noiselevels = [0, 10, 25, 50]
    dropoutlevels = [0, 1, 2, 3]
    time_thresholds.reverse()
    distance_thresholds.reverse()
    # noiselevels.reverse()
    # dropoutlevels.reverse()


    data_dir = "data/v1"
    for func in funcs:
        for time_thresh in time_thresholds:
            for dist_thresh in distance_thresholds:
                for noiselevel in noiselevels:
                    for dropoutlevel in dropoutlevels :
                        id=f"noiselevel{noiselevel}_dropoutlevel{dropoutlevel}"
                        input_path=f"{data_dir}/trajectories_{id}.parquet"
                        input_df = get_df(input_path)
                        input_df = get_processed_trajectory_df(input_df)
                        time_thresh_min = time_thresh
                        dist_thresh_m = dist_thresh
                        output_path=f"{data_dir}/detected_staypoints_{id}/algorithm_{func.__name__}/time_threshold_{time_thresh_min}_minutes_dist_threshold_{dist_thresh_m}_meters.parquet"
                        print(f"Approach: {func.__name__}, time_thresh_min: {time_thresh_min}, dist_thresh: {dist_thresh_m} for noiselevel: {noiselevel}, dropoutlevel: {dropoutlevel}")
                    
                        try:
                            calculate_stay_points(func=func,
                                                input_df=input_df,
                                                output_path=output_path, 
                                                time_thresh_min=time_thresh_min,                
                                                dist_thresh_m=dist_thresh_m)
                            evaluate(calculated_data_path=output_path, ground_truth_path=f"{data_dir}/ground_truth.parquet")
                        except Exception as e:
                            print(f"Error processing approach: {func.__name__}, time_thresh: {time_thresh_min}, dist_thresh: {dist_thresh_m} for noiselevel: {noiselevel}, dropoutlevel: {dropoutlevel}. Error: {e}")
                       