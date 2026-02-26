# This file contains the evaluation code for scoring the stay point detection results against the ground truth data. The main function is `get_score` which takes in the ground truth and calculated stay points as pandas DataFrames, and computes precision, recall, F1 score, and F2 score based on spatial and temporal proximity criteria. The evaluation is done by checking how many stay points in the calculated results match with the ground truth within a specified radius (r) and time window (t).

# Authors: 
# Hossein Amiri (hossein.amiri@emory.edu)
# Lance Kennedy (lance.kennedy@emory.edu)

from concurrent.futures import ProcessPoolExecutor

import pandas


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

    with ProcessPoolExecutor() as ex:
        precision_score = ex.submit(get_precision_score, gt, calc, r, t).result()
        recall_score = ex.submit(get_recall_score, gt, calc, r, t).result()
    f1 = 2*(precision_score * recall_score) / (precision_score + recall_score + 1e-10)  # F1 score
    f2 = 5*(precision_score * recall_score) / (4*precision_score + recall_score + 1e-10)  # F2 score


    score = {
        'precision': float(precision_score),
        'recall': float(recall_score),
        'f1': float(f1),
        'f2': float(f2)
    }
    return score

def get_recall_score(gt, calc, r, t):
    print("Calculating recall score...")
    recall_score = get_match_score(gt, calc, r, t)
    return recall_score

def get_precision_score(gt, calc, r, t):
    print("Calculating precision score...")
    precision_score = get_match_score(calc, gt, r, t)
    return precision_score

def get_match_score(df1, df2, r, t, chunk_size=1000):
    print("Calculating match score...")
    df1['matched'] = False
    chunks = [df1[i:i+chunk_size] for i in range(0, len(df1), chunk_size)]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(get_match_score_chunk, chunks, [df2]*len(chunks), [r]*len(chunks), [t]*len(chunks)))
    df1 = pandas.concat(results)
    match_score = df1['matched'].mean()
    return match_score

def get_match_score_chunk(df1_chunk, df2, r, t):
    for idx, row in df1_chunk.iterrows():
        agent_id = row['agent_id']
        lat = row['latitude']
        lon = row['longitude']
        arrive_time = row['arrive_time']
        leave_time = row['leave_time']

        matches = df2[
            (df2['agent_id'] == agent_id) &
            (df2['latitude'].between(lat - r, lat + r)) &
            (df2['longitude'].between(lon - r, lon + r)) &
            (df2['arrive_time'].between(arrive_time - t, arrive_time + t)) &
            (df2['leave_time'].between(leave_time - t, leave_time + t))
        ]
        if not matches.empty:
            df1_chunk.at[idx, 'matched'] = True
    return df1_chunk