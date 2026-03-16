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

# Leader Board (Dataset V1: POL Simulated Trajectories)

| Rank | Team Name                                              | F1 Score                          |
| ---- | ------------------------------------------------------ | --------------------------------- |
| 1    | Riyang_M8_HistGradientBoosting_Classifier (Supervised) | 0.9789 Test set only (100 agents) |
| 1    | Hossein_Copilot_ChatGPT.py                             | 0.9687746641339697                |
| 3    | Riyang_M7_Hidden_Markov_Model                          | 0.9543                            |
| 4    | Lance_Temporal_DBSCAN                                  | 0.9521875531303055                |
| 5    | Adaptive_sliding_window                                | 0.9470605843790482                |
| 6    | Baseline_Similarity_Paper.py                           | 0.9129668240132905                |
| 7    | Alex_Temporal_DBSCAN                                   | 0.8540195286190208                |
| 8    | Baseline_Track_Intel                                   | 0.8206979896924229                |

# Leader Board (Dataset V2: Sim1 Simulated Trajectories)

| Rank | Team Name                                              | F1 Score                          |
| ---- | ------------------------------------------------------ | --------------------------------- |
| 1    | Lance_Temporal_DBSCAN                                  | 0.5242599940960189                |
| 2    | Baseline_Track_Intel                                   | 0.1327184469412969                |

# Methods:

### Riyang_M8_HistGradientBoosting_Classifier (Supervised)

Implementation in [code/approaches/m8.py](code/approaches/m8.py)

A supervised approach that serves as a performance ceiling. We engineer 23 features per trajectory point — including forward/backward displacement, multi-scale rolling statistics (std, mean, max over windows of 5/15/30/60 points), distance to rolling centroid, and hour of day — then train a gradient boosting classifier on 80% of agents and evaluate on the remaining 20%.

### Hossein_Copilot_ChatGPT.py

Implementation in [code/approaches/b3.py](code/approaches/b3.py)

- `haversine_m` computes the great-circle distance in meters between two latitude/longitude coordinates using the Haversine formula.
- `b3` processes time-sorted location records for each agent and identifies **stay points** where the agent remains within a spatial radius (`dist_thresh_m`, default 200 m) for at least a minimum duration (`time_thresh_min`, default 20 minutes).
- When such a period is found, it outputs a summarized stay point containing the mean location, arrival and leave times, duration, number of points, and the maximum radius from the center.

The method follows the classic **Li et al. stay point detection approach** by expanding a window from point _i_ until the distance threshold is exceeded, then checking if the time spent within that region satisfies the temporal threshold.

### Riyang_M7_Hidden_Markov_Model

Implementation in [code/approaches/m7.py](code/approaches/m7.py)

An unsupervised approach that models each trajectory point as either "stopped" or "moving" using a 2-state HMM. The observation at each point is the haversine displacement to the next point, modeled with Gaussian emission distributions for each state. We decode the optimal state sequence via the Viterbi algorithm and merge consecutive "stopped" segments into staypoints (filtered by a minimum duration threshold).

### Lance_Temporal_DBSCAN

Implementation in [code/approaches/lance.py](code/approaches/lance.py)

### Baseline_Similarity_Paper.py

Implementation in [code/approaches/b2.py](code/approaches/b2.py)

Implements **Li et al. (2008) stay point detection** for each agent’s time-ordered GPS trace.

- `getDistance` computes the great-circle distance (meters) between two lon/lat points using the Haversine formula.
- `b2` scans each agent’s records with two indices (`i`, `j`), expanding forward until the distance from point `i` to point `j` exceeds `dist_thresh_m` (default 200 m).
- If the elapsed time between `time[i]` and `time[j]` exceeds `time_thresh_min` (default 20 minutes), it emits a stay point summary for the window `i..j` using mean latitude/longitude plus arrival time, leave time, duration, and number of samples.
- After emitting, it jumps the start index to `j` (then advances) and continues searching for additional stay points.

### Alex_Temporal_DBSCAN

Implementation in [code/approaches/alex.py](code/approaches/alex.py)

### Baseline_Track_Intel

Implementation in [code/approaches/b1.py](code/approaches/b1.py)
