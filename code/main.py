import pyarrow.parquet as parquet
import pandas
import approaches.master as master
from approaches.b2 import b2 
from approaches.b3 import b3 
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

def save_df(df, output_path = "data/v1/b2/sp1.csv"):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .parquet")
        
def calculate_stay_points(func=None, input_path = "data/v1/trajectory.parquet", output_path = "data/v1/b2/sp2.csv", **kwargs):
    if os.path.exists(output_path):
        print(f"Output already exists at {output_path}. Skipping calculation.")
        return
    df = get_df(input_path)
    sdf = master.get_stay_points(func=func, df=df, **kwargs)
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
    funcs=[b3, b2]
    time_thresholds = [100, 50, 25, 10, 5, None]
    distance_thresholds = [500, 450, 400, 350, 300, 250, 200, 150, 100, 50, None]
    noiselevels = [0, 10, 25, 50]
    noiselevels.reverse()
    dropoutlevels = [0, 1, 2, 3]
    dropoutlevels.reverse()


    data_dir = "data/v1"
    for func in funcs:
        for time_thresh in time_thresholds:
            for dist_thresh in distance_thresholds:
                for noiselevel in noiselevels:
                    for dropoutlevel in dropoutlevels :
                        id=f"nl{noiselevel}_dl{dropoutlevel}"
                        output_path=f"{data_dir}/{id}/{func.__name__}/{time_thresh}_{dist_thresh}.parquet"
                        
                        print(f"Approach: {func.__name__}, time_thresh: {time_thresh}, dist_thresh: {dist_thresh} for noiselevel: {noiselevel}, dropoutlevel: {dropoutlevel}")
                        try:
                            calculate_stay_points(func=func,
                                                input_path=f"{data_dir}/trajectories_noiselevel{noiselevel}_dropoutlevel{dropoutlevel}.parquet",
                                                output_path=output_path, 
                                                time_thresh_min=time_thresh,                
                                                dist_thresh_m=dist_thresh)
                            evaluate(calculated_data_path=output_path, ground_truth_path=f"{data_dir}/ground_truth.parquet")
                        except Exception as e:
                            print(f"Error processing approach: {func.__name__}, time_thresh: {time_thresh}, dist_thresh: {dist_thresh} for noiselevel: {noiselevel}, dropoutlevel: {dropoutlevel}. Error: {e}")
                       