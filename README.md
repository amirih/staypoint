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
| 1    | BaseLine 1 | 0.8206979896924229 |
| 2    | Baseline 2 | 0.7168635343899474 |
| 3    | Team A     | 0.7118115823982403 |
