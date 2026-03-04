Please note that I renamed the following files as the input:

- naive_sps.parquet -> ground_truth.parquet
- noised_trajectory.parquet -> trajectory.parquet

I Also renamed these columns in the code from ground_truth.parquet

- startTime -> arrive_time
- endTime -> leave_time

Dropped the following columns in the code from trajectory.parquet

- longitude
- latitude

Also renamed these columns in the code from trajectory.parquet

- n_lon -> longitude
- n_lat -> latitude

# Leader Board

| Rank | Team Name  | F1 Score           |
| ---- | ---------- | ------------------ |
| 1    | M8 (Supervised) | 	0.9789 Test set only (100 agents)|
| 2    | Baseline 3 | 0.9687746641339697 |
| 3    | M7 (HMM)   | 0.9543             |
| 4    | Lance - temporal DBSCAN    | 0.9521875531303055 |
| 5    | Baseline 2 | 0.9129668240132905 |
| 6    | Alex - temporal DBSCAN     | 0.8540195286190208 |
| 7    | BaseLine 1 (trackintel) | 0.8206979896924229 |
