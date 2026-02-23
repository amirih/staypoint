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
