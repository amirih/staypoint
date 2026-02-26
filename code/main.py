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

def get_input_data(data_path = "data/v1/trajectory.parquet"):
    df = parquet.read_table(data_path).to_pandas()
    return df

def save_output_data(df, file_name = "stay_points.csv",output_path = "data/v1/b2/sp1.csv"):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

        
def calculate_stay_points(func=None,output_path = "data/v1/b2/sp2.csv", **kwargs):
    df = get_input_data()
    sdf = master.get_stay_points(func=func, df=df, **kwargs)
    save_output_data(sdf, output_path=output_path)

def evaluate(output_path = "data/v1/b2/sp2.csv"):
    output_dir = os.path.dirname(output_path)
    file_name = os.path.basename(output_path)
    gt_df = parquet.read_table("data/v1/ground_truth.parquet").to_pandas()
    gt_df.rename(columns={'startTime': 'arrive_time', 'endTime': 'leave_time'}, inplace=True)
    calc_df = pandas.read_csv(output_path)
    print("Calculating evaluation score...")
    score = eval_utils.get_score(gt_df, calc_df)
    utils.save_json(score, os.path.join(output_dir, f"{file_name}_score.json"))
    print(f"Evaluation Score: {score}")

if __name__ == "__main__":
    funcs=[b3, b2]
    time_thresholds = [5, 10, 15, 20]  
    distance_thresholds = [50, 100, 150, 200]
    for func in funcs:
        for time_thresh in time_thresholds:
            for dist_thresh in distance_thresholds:
                print(f"Running approach: {func.__name__}, time_thresh: {time_thresh}, dist_thresh: {dist_thresh}")
                output_path=f"data/v1/{func.__name__}/{time_thresh}_{dist_thresh}.csv"
                calculate_stay_points(func=func,
                                      output_path=output_path, 
                                      time_thresh_min=time_thresh,                dist_thresh_m=dist_thresh)
                evaluate(output_path=output_path)