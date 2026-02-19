
import pandas


def get_score(ground_truth_df, calculated_df, r=0.001):

    common_columns = ['agent_id','latitude','longitude','arrive_time','leave_time']
    gt = ground_truth_df[common_columns].copy()
    calc = calculated_df[common_columns].copy()

    gt['arrive_time'] = pandas.to_datetime(gt['arrive_time'])
    gt['leave_time'] = pandas.to_datetime(gt['leave_time'])
    calc['arrive_time'] = pandas.to_datetime(calc['arrive_time'])
    calc['leave_time'] = pandas.to_datetime(calc['leave_time'])

    # score how many stay points are correctly identified
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
            (calc['arrive_time'] <= leave_time) &
            (calc['leave_time'] >= arrive_time)
        ]
        if not matches.empty:
            gt.at[idx, 'matched'] = True
    score = gt['matched'].mean()
    return score


