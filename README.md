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
| 1    | Baseline 3 | 0.968468847171057  |
| 2    | Team A     | 0.9521875531303055 |
| 3    | Team B     | 0.8540195286190208 |
| 4    | BaseLine 1 | 0.8206979896924229 |
| 5    | Baseline 2 | 0.7168635343899474 |
