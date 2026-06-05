# Dataset Download

The datasets are available at [https://osf.io/v8e4g/](https://osf.io/v8e4g/).

# Project Structure

# Experimental Results
### Metrics
- Hyperparameter dependent metrics, calculated with the matched predictions based on two thresholds, *r* the GPS distance and *t* the time difference
  - Precision
  - Recall
  - F1
- Hyperparameter independent metrics
  - Temporal IoU. Intersect of prediction and ground truth time durations divided by their Union
  - Gps distance. If no match then *Inf*

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


### ASW

Implementation in [code/approaches/mo.py](code/approaches/mo.py) --> Function `asw()`

- This approach is a modified approach of the original sliding window approach.
- Instead of using a constant stay radius value, `asw()` computes it locally and adjusts the stay radius based on local spatial variability of the trajectory.

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

### Hybrid Confirmation Filter (Steve)

Implementation in [code/hybrid.py](code/hybrid.py). Run all 16 noise/dropout combinations with [code/run_three_hybrids.py](code/run_three_hybrids.py).

An unsupervised ensemble method that takes a primary algorithm's staypoints and keeps only those confirmed by at least one other algorithm within ±0.001° and ±5 minutes. Confirmation is logit-weighted by each confirmer's precision — higher-precision algorithms contribute more evidence. Threshold T=0.90 (any single confirmation suffices) was validated across all noise/dropout levels.

Three variants:

| Variant | Primary | Confirmers | Output folder |
|---------|---------|-----------|---------------|
| Unsupervised | HMM-GEM | HSW, T-DBSCAN | `hybrid_unsupervised/` |
| Optimized | SSPE | HMM-GEM, Trackintel, HSW, T-DBSCAN sweep, ASW | `hybrid_optimized/` |
| Fully Supervised | SSPE | same + Gradient Boosting | `hybrid_supervised/` |

Results at noise=25, dropout=2 (default eval r=0.001, t=5):

| Variant | F1 | Precision | Recall |
|---------|-----|-----------|--------|
| hybrid_unsupervised | 0.7773 | 0.8461 | 0.7188 |
| hybrid_optimized | **0.7946** | **0.8396** | 0.7542 |
| hybrid_supervised | 0.7946 | 0.8396 | 0.7543 |
| SSPE (best individual) | 0.7935 | 0.8348 | 0.7561 |

hybrid_optimized beats SSPE at every noise/dropout level tested. The gain is largest under heavy noise (NL=50): +0.002 to +0.003 F1 over SSPE, and +0.04 to +0.10 over HMM-GEM alone.
