
import pandas
from utils import print_time as print


def get_score(ground_truth_df, calculated_df, r=0.001, t=5):
    t = pandas.Timedelta(minutes=t)
    common_columns = ['agent_id','latitude','longitude','arrive_time','leave_time']
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

    score = (recall_score+precision_score)/2
    return score


