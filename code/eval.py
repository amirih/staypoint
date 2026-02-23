# This file contains the evaluation code for scoring the stay point detection results against the ground truth data. The main function is `get_score` which takes in the ground truth and calculated stay points as pandas DataFrames, and computes precision, recall, F1 score, and F2 score based on spatial and temporal proximity criteria. The evaluation is done by checking how many stay points in the calculated results match with the ground truth within a specified radius (r) and time window (t).

# Authors: 
# Hossein Amiri (hossein.amiri@emory.edu)
# Lance Kennedy (lance.kennedy@emory.edu)

import pandas
from utils import print_time as print


def get_score(ground_truth_df, calculated_df, r=0.001, t=5):     
    t = pandas.Timedelta(minutes=t)
    common_columns = ['agent_id','latitude','longitude','arrive_time','leave_time']
    assert all(col in ground_truth_df.columns for col in common_columns), f"Ground truth data missing required columns: {common_columns}"
    assert all(col in calculated_df.columns for col in common_columns), f"Calculated data missing required columns: {common_columns}"
    gt = ground_truth_df[common_columns].copy()
    calc = calculated_df[common_columns].copy()

    gt['arrive_time'] = pandas.to_datetime(gt['arrive_time'])
    gt['leave_time'] = pandas.to_datetime(gt['leave_time'])
    calc['arrive_time'] = pandas.to_datetime(calc['arrive_time'])
    calc['leave_time'] = pandas.to_datetime(calc['leave_time'])

    # score how many stay points are correctly identified
    print("Calculating recall score...")
    gt['matched'] = False
    for idx, row in gt.iterrows():
        agent_id = row['agent_id']
        lat = row['latitude']
        lon = row['longitude']
        arrive_time = row['arrive_time']
        leave_time = row['leave_time']

        matches = calc[
            (calc['agent_id'] == agent_id) &
            (calc['latitude'].between(lat - r, lat + r)) &
            (calc['longitude'].between(lon - r, lon + r)) &
            (calc['arrive_time'].between(arrive_time - t, arrive_time + t)) &
            (calc['leave_time'].between(leave_time - t, leave_time + t))
        ]
        if not matches.empty:
            gt.at[idx, 'matched'] = True
    recall_score = gt['matched'].mean()

    # score how many identified stay points are in GT
    print("Calculating precision score...")
    calc['matched'] = False
    for idx, row in calc.iterrows():
        agent_id = row['agent_id']
        lat = row['latitude']
        lon = row['longitude']
        arrive_time = row['arrive_time']
        leave_time = row['leave_time']

        matches = gt[
            (gt['agent_id'] == agent_id) &
            (gt['latitude'].between(lat - r, lat + r)) &
            (gt['longitude'].between(lon - r, lon + r)) &
            (gt['arrive_time'].between(arrive_time - t, arrive_time + t)) &
            (gt['leave_time'].between(leave_time - t, leave_time + t))
        ]
        if not matches.empty:
            calc.at[idx, 'matched'] = True
    precision_score = calc['matched'].mean()

    f1 = 2*(precision_score * recall_score) / (precision_score + recall_score + 1e-10)  # F1 score
    f2 = 5*(precision_score * recall_score) / (4*precision_score + recall_score + 1e-10)  # F2 score


    score = {
        'precision': float(precision_score),
        'recall': float(recall_score),
        'f1': float(f1),
        'f2': float(f2)
    }
    return score


