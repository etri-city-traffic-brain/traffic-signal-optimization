# RL-constraints-traffic
Decentralized Deep Reinforcement Learning based Real-World Applicable Traffic Signal Optimization


## Decentralized DQN 

### Prerequisite
- python 3.7.9 above
- pytorch 1.7.1 above
- tensorboard 2.0.0 above

### How to use
check the condition state (throughput)
```shell script
    python run.py simulate --network [5x5grid, 5x5grid_v2, dunsan, dunsan_v2]
``` 
"Traffic data of Dunsan and Dunsan_v2 are classified by government of South Korea."

Run in RL algorithm DQN (default device: cpu)
```shell script
    python run.py train --network [5x5grid, 5x5grid_v2, dunsan, dunsan_v2]
``` 
"Traffic data of Dunsan and Dunsan_v2 are classified by government of South Korea."

- check the result
Tensorboard
```shell script
    tensorboard --logdir ./training_data
``` 
Hyperparameter in json, model is in `./training_data/[time you run]/model` directory.

- replay the model
```shell script
    python run.py test --replay_name /replay_data in training_data dir/ --replay_epoch NUM
```
### Performance
Synthetic Data in 5x5grid(Straight Flow), 5x5grid_v2(Random Trips)</br>
|Evaluation Metric|Method|Straight Flow|Random Trips|
|:---:|:---:|---:|---:|
|Throughput|Fixed|15184|6872|
|Throughput|Ours|**15380**|**7154**|
|Waiting Time|Fixed|916.7|670.2|
|Waiting Time|Ours|**851.2**|**587.1**|

Real World Data in Dunsan-dong, Daejeon, Korea
|Evaluation Metric|Method|Peak|Free and Peak combined|
|:---:|:---:|---:|---:|
|Throughput|Fixed|13616|10415|
|Throughput|Ours|**13649**|**10539**|
|Waiting Time|Fixed|240.1|237.6|
|Waiting Time|Ours|**231.1**|**218.1**|
