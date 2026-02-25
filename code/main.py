import pyarrow.parquet as parquet
import pandas
import approaches.b2 as b2
import eval as eval_utils
import os
from utils import print_time as print
import utils 

def get_input_data(data_path = "data/v1/trajectory.parquet"):
    df = parquet.read_table(data_path).to_pandas()
    return df

def save_output_data(df, file_name = "stay_points.csv",output_path = "data/v1/b2/sp1.csv"):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

        
def calculate_stay_points(output_path = "data/v1/b2/sp2.csv"):
    df = get_input_data()
    sdf = b2.get_stay_points(df)
    save_output_data(sdf,output_path=output_path)
    print(sdf.head())

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
    output_path="data/v1/b2/sp2.csv"
    calculate_stay_points(output_path=output_path)
    evaluate(output_path=output_path)