# Dataset Download

The datasets are available at [https://osf.io/v8e4g/](https://osf.io/v8e4g/).

# Project Structure

# Experimental Results

# Environmental Setup

# Reproducing the Results

### Hanqi_Hyperband_Trackintel

POL Implementation in [code/approaches/POL_Hyperband_Verification.ipynb](code/approaches/POL_Hyperband_Verification.ipynb)

SIM1 Implementation in [code/approaches/SIM1_Hyperband_Verification.ipynb](code/approaches/SIM1_Hyperband_Verification.ipynb)


The Hyperband method is used to explore and optimize parameters in the Trackintel search space. In our setting, there are three parameters: *time*, *distance*, and *gap*. The search space we use is:

- *time:* 5–20
- *distance:* 20–200
- *gap:* 30–120

We design the process in *three stages*.

In *Stage 1*, we use *25% of the POL data* and *2 shards of SIM1*. Since SIM1 is large, we divide it into *10 shard files* for easier loading; each shard contains *100 users*, except the last shard, which contains *33 users*. We first randomly generate *36 candidate combinations* with *random seed 13*. These candidates are evaluated and ranked, and then we select the *top 12* for iterative refinement. In each iteration, we keep the *top 2 combinations*, while the other *10 combinations* are regenerated based on the parameter range covered by the current top 12 candidates, which is smaller than the original search space. After *10 iterations for POL* and *4 iterations for SIM1*, the method moves to Stage 2.

In *Stage 2*, we evaluate using *75% of the POL data* and *5 shards of SIM1*. From this stage, we select the *top 4 combinations* and continue iterative refinement. In each iteration, we keep the *top 1 combination* and regenerate the other *3 combinations*. This stage also runs for *10 iterations for POL* and *4 iterations for SIM1*, and then proceeds to Stage 3.

In *Stage 3*, we evaluate the remaining candidates on the *full dataset*, using *100% of POL* and *all 10 shards of SIM1*. We also perform *5 iterations for POL* and *4 iterations for SIM1*. The final *top 1 combination* is selected as the *locally optimized parameter setting*.


### Riyang_M8_HistGradientBoosting_Classifier (Supervised)

Implementation in [code/approaches/m8.py](code/approaches/m8.py)

A supervised approach that serves as a performance ceiling. We engineer 23 features per trajectory point — including forward/backward displacement, multi-scale rolling statistics (std, mean, max over windows of 5/15/30/60 points), distance to rolling centroid, and hour of day — then train a gradient boosting classifier on 80% of agents and evaluate on the remaining 20%.

### HSW

Implementation: [code/approaches/hsw.py](code/approaches/hsw.py)


### Mo_adaptive_sliding_window

Implementation in [code/approaches/b3.py](code/approaches/b3.py) --> Function `b3_adaptive()`

- This approach is a modified approach of Hossein's baselinse approach.
- `b3` uses a constant stay radius, while `b3_adaptive` adjusts the stay radius based on local spatial variability of the trajectory.

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
