cd ~/onone/onnuc/Research/staypoint/data/v1/detected_staypoints_noiselevel50_dropoutlevel3/algorithm_hsw
grep -r '"f1": 0.9' .
grep -r '"f1": 0.8' .
grep -r '"f1": 0.7' .
grep -r '"f1": 0.6' .
grep -r '"f1": 0.5' .

cp time_threshold_15_minutes_dist_threshold_200_meters.parquet staypoints.parquet
cat time_threshold_15_minutes_dist_threshold_200_meters.parquet_score.json 
