import pyarrow.parquet as parquet
import pandas
import m1
import eval as eval_utils
import os
from utils import print_time as print

def get_input_data(data_path = "data/v1/trajectory.parquet"):
    df = parquet.read_table(data_path).to_pandas()
    return df

def save_output_data(df, file_name = "stay_points.csv",output_dir = "data/v1/m1/"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=False)

        
def calculate_stay_points():
    df = get_input_data()
    sdf = m1.get_stay_points(df)
    save_output_data(sdf,"sp2.csv")
    print(sdf.head())

def evaluate():
    gt_df = parquet.read_table("data/v1/ground_truth.parquet").to_pandas()
    gt_df.rename(columns={'startTime': 'arrive_time', 'endTime': 'leave_time'}, inplace=True)
    calc_df = pandas.read_csv("data/v1/m1/sp2.csv")
    print("Calculating evaluation score...")
    score = eval_utils.get_score(gt_df, calc_df)
    print(f"Evaluation Score: {score:.4f}")

if __name__ == "__main__":
    calculate_stay_points()
    evaluate()