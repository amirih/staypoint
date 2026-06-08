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

### Results on noise level 25 dropout level 2
*Evaluaiton results on difference noise levels and dropout levels could be find in folder [data/v1/evaluation_summary](data/v1/evaluation_summary)*

#### Evaluate with r = 0.001, t = 5
| approach                          | f1                   | precision            | recall              | temporal_IoU         | GPS_distance           |
|-----------------------------------|----------------------|----------------------|---------------------|----------------------|------------------------|
| algorithm_centroid_sliding_window | 0.5228928382399001   | 0.48745439752237474  | 0.5638880876838765  | 0.7092791151256903   | 0.0003456343698450831  |
| algorithm_gradient_boosting       | 0.24937939592012384  | 0.840728794350972    | 0.14640292684407877 | 0.9444748981517632   | 0.0002492470140832113  |
| algorithm_hmm-gem                 | 0.7622701080880542   | 0.7939556374040153   | 0.7330165621726655  | 0.8878156611093382   | 2.9584463007463112e-05 |
| algorithm_hsw                     | 0.7335107285920084   | 0.7849674560310206   | 0.688385226116922   | 0.9071081147335814   | 3.8319585955904264e-05 |
| asw                               | 0.5589356153814123   | 0.621757804510719    | 0.5076434958177098  | 0.7720811795920306   | 8.52061250929918e-05   |
| sspe                              | 0.7934838889053939   | 0.834785960913144    | 0.7560760858014665  | 0.9255240021622453   | 2.8803362882222317e-05 |
| temporal-dbscan                   | 0.691741829020066    | 0.7116682429508039   | 0.6729008850363578  | 0.8398141755579325   | 3.9954468120287756e-05 |
| temporal-dbscan-sweep             | 0.7291210409159219   | 0.7732569018013574   | 0.6897514915063835  | 0.89053539119458     | 3.93382899874331e-05   |
| ti_baseline                       | 0.018708888862310053 | 0.010206919068179398 | 0.11200340048274711 | 0.05366436349973034  | 0.00010102406417739497 |
| ti_dropout_filter                 | 0.01870468529510015  | 0.010205678778753682 | 0.11185159321725138 | 0.053642300287047515 | 0.00010066420785504712 |
| Trackintel_Hyperband_Search       | 0.7477138070029642   | 0.8097960773206826   | 0.6944726974633006  | 0.8877456857285236   | 0.00010242936247157263 |


#### Fixed t = 5, and evaluate with difference R's, on noise level 25 dropout level 2
| approach                          | f1                   | f1                   | f1                   | f1                   | precision            | precision            | precision            | precision            | recall              | recall              | recall              | recall              |
|-----------------------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|---------------------|---------------------|---------------------|---------------------|
|                                   | r 0.0001             | r 0.0005             | r 0.001              | r 0.01               | r 0.0001             | r 0.0005             | r 0.001              | r 0.01               | r 0.0001            | r 0.0005            | r 0.001             | r 0.01              |
| algorithm_centroid_sliding_window | 0.2531620622413077   | 0.4227203941082794   | 0.5228928382399001   | 0.56732007737416     | 0.23600430434896721  | 0.3940710217579591   | 0.48745439752237474  | 0.5288706333167108   | 0.2730101862675148  | 0.45586203755711746 | 0.5638880876838765  | 0.6117984606743279  |
| algorithm_gradient_boosting       | 0.24451799749232075  | 0.24937939592012384  | 0.24937939592012384  | 0.24937939592012384  | 0.8243396390898788   | 0.840728794350972    | 0.840728794350972    | 0.840728794350972    | 0.1435489502527591  | 0.14640292684407877 | 0.14640292684407877 | 0.14640292684407877 |
| algorithm_hmm-gem                 | 0.7438787591260209   | 0.7622701080880542   | 0.7622701080880542   | 0.7622701080880542   | 0.7747998092638572   | 0.7939556374040153   | 0.7939556374040153   | 0.7939556374040153   | 0.7153310157424134  | 0.7330165621726655  | 0.7330165621726655  | 0.7330165621726655  |
| algorithm_hsw                     | 0.7045559320439378   | 0.7334783768640217   | 0.7335107285920084   | 0.7335107285920084   | 0.7539814430134331   | 0.7849328347874256   | 0.7849674560310206   | 0.7849674560310206   | 0.6612117255931869  | 0.6883548646638228  | 0.688385226116922   | 0.688385226116922   |
| asw                               | 0.5447616500140259   | 0.5587517549815996   | 0.5589356153814123   | 0.5590024737086171   | 0.6059907405685812   | 0.62155327891713     | 0.621757804510719    | 0.6218321774538423   | 0.49477023970367223 | 0.5074765078256646  | 0.5076434958177098  | 0.5077042187239081  |
| sspe                              | 0.7771697135758044   | 0.7934838889053939   | 0.7934838889053939   | 0.7934838889053939   | 0.8176226073547652   | 0.834785960913144    | 0.834785960913144    | 0.834785960913144    | 0.7405310218147041  | 0.7560760858014665  | 0.7560760858014665  | 0.7560760858014665  |
| temporal-dbscan                   | 0.6778427191251065   | 0.6916794537317639   | 0.691741829020066    | 0.691741829020066    | 0.6969847554622253   | 0.7116041229181027   | 0.7116682429508039   | 0.7116682429508039   | 0.6597240143913288  | 0.6728401621301595  | 0.6729008850363578  | 0.6729008850363578  |
| temporal-dbscan-sweep             | 0.7157557609663783   | 0.7290568673866139   | 0.7291210409159219   | 0.7291210409159219   | 0.758951504533161    | 0.7731888618619129   | 0.7732569018013574   | 0.7732569018013574   | 0.6772122113764365  | 0.6896907686001852  | 0.6897514915063835  | 0.6897514915063835  |
| ti_baseline                       | 0.014129293672152885 | 0.018708888862310053 | 0.018708888862310053 | 0.0187186082541653   | 0.007708451212780645 | 0.010206919068179398 | 0.010206919068179398 | 0.010212452773285485 | 0.08458700833421888 | 0.11200340048274711 | 0.11200340048274711 | 0.11203376193584624 |
| ti_dropout_filter                 | 0.01414021404263767  | 0.01870468529510015  | 0.01870468529510015  | 0.018711876573106544 | 0.007715205048542076 | 0.010205678778753682 | 0.010205678778753682 | 0.010209834185422557 | 0.08455664688111973 | 0.11185159321725138 | 0.11185159321725138 | 0.11186677394380096 |
| Trackintel_Hyperband_Search       | 0.7318596003262184   | 0.7477138070029642   | 0.7477138070029642   | 0.7477301515459299   | 0.7926255044962118   | 0.8097960773206826   | 0.8097960773206826   | 0.8098137789421511   | 0.6797473927102151  | 0.6944726974633006  | 0.6944726974633006  | 0.6944878781898501  |

#### Fixed r = 0.001, and evaluate with difference T's, on noise level 25 dropout level 2
| approach                          | f1                     | f1                   | f1                  | precision              | precision            | precision            | recall                | recall              | recall              |
|-----------------------------------|------------------------|----------------------|---------------------|------------------------|----------------------|----------------------|-----------------------|---------------------|---------------------|
|                                   | t = 1                  | t = 5                | t = 10              | t = 1                  | t = 5                | t = 10               | t = 1                 | t = 5               | t = 10              |
| algorithm_centroid_sliding_window | 0.002702797768318118   | 0.5228928382399001   | 0.5355401761237251  | 0.0025196189076402194  | 0.48745439752237474  | 0.4993176032125141   | 0.002914699497517951  | 0.5638880876838765  | 0.5774292957660954  |
| algorithm_gradient_boosting       | 0.24247517581255243    | 0.24937939592012384  | 0.252172114165883   | 0.8174527068259088     | 0.840728794350972    | 0.8501438409903235   | 0.14234967285534286   | 0.14640292684407877 | 0.1480424453114326  |
| algorithm_hmm-gem                 | 0.05725787349968892    | 0.7622701080880542   | 0.7716315415082136  | 0.05963793018399461    | 0.7939556374040153   | 0.8037062005689198   | 0.05506049519530005   | 0.7330165621726655  | 0.7420187330165622  |
| algorithm_hsw                     | 0.6562548021598378     | 0.7335107285920084   | 0.7509968375688042  | 0.7022919263259936     | 0.7849674560310206   | 0.8036802381941559   | 0.615882076116163     | 0.688385226116922   | 0.70479559151701    |
| asw                               | 0.5174333087691237     | 0.5589356153814123   | 0.5783553190949393  | 0.5755908000669356     | 0.621757804510719    | 0.643493297138501    | 0.46994975179512094   | 0.5076434958177098  | 0.5251924157090159  |
| sspe                              | 0.7289759827437655     | 0.7934838889053939   | 0.8039988847232933  | 0.7669203177902182     | 0.834785960913144    | 0.845848278636318    | 0.6946093240022467    | 0.7560760858014665  | 0.7660953653241844  |
| temporal-dbscan                   | 0.6506968994641177     | 0.691741829020066    | 0.7090067552390698  | 0.6689643011717936     | 0.7116682429508039   | 0.7304233525159097   | 0.6334006345543698    | 0.6729008850363578  | 0.68881028646031    |
| temporal-dbscan-sweep             | 0.6840246931359735     | 0.7291210409159219   | 0.7431178043191699  | 0.7252547245232952     | 0.7732569018013574   | 0.7884298082974707   | 0.6472302764410305    | 0.6897514915063835  | 0.7027310127062681  |
| ti_baseline                       | 0.002657483793818983   | 0.018708888862310053 | 0.0345658804438841  | 0.0014498307377950675  | 0.010206919068179398 | 0.018858867001548055 | 0.01590940142395215   | 0.11200340048274711 | 0.20682221851137794 |
| ti_dropout_filter                 | 0.0026604926835523913  | 0.01870468529510015  | 0.03456693556636293 | 0.001451622062993195   | 0.010205678778753682 | 0.018861390870017494 | 0.01590940142395215   | 0.11185159321725138 | 0.20659450761313436 |
| Trackintel_Hyperband_Search       | 0.00013075629401962654 | 0.7477138070029642   | 0.7596126342820063  | 0.00014161297174821214 | 0.8097960773206826   | 0.8226828577497699   | 0.0001214458123965813 | 0.6944726974633006  | 0.7055242663913895  |

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

### Temporal_DBSCAN

Implementation in [code/approaches/temporal_dbscan.py](code/approaches/temporal_dbscan.py)

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

Implementation in [code/approaches/ti_baseline.py](code/approaches/ti_baseline.py)

### Trackintel_Dropout_Filter

Implementation in [code/approaches/ti_dropout_filter.py](code/approaches/ti_dropout_filter.py)
