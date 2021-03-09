# city-traffic_RL
City Optimization Reinforcement Learning based on https://github.com/3neutronstar/traffic-optimization_RL

## Decentralized DQN 
- Experiment
    1) Every 160s(depend on tl_phase_period, Update asynchronously)
    2) Controls the each phase length that phases are in intersection system

- Agents
    1) Traffic Light Systems (Intersection)
    2) Have their own offset value
    3) Update itself asynchronously (according to offset value and COMMON_PERIOD value)

- State
    1) Queue Length(2 spaces per each inEdge, total '2*n' spaces when inEdge is n) <br/>
    -> each number of vehicle is divided by max number of vehicles in an edge.(Normalize, TODO)
    2) Phase Length(If the number of phase is 4, spaces are composed of 4) <br/>
    -> (up,right,left,down) is divided by max period (Normalize, TODO)
    3) Searching method
        (1) Before phase ends, receive all the number of inflow vehicles(not in 'all red', 'all yellow' phase)

- Action (per each COMMON_PERIOD of intersection)
    1) Tuple of +,- of each phases (18) <- 4 phases, (7) <- 3 phases, (2) <- 2 phases (over 5, TODO)
    2) Length of phase time changes
    -> minimum value exists and maximum value exists (currently fixed by 4)

- Next State
    1) For agent, next state will be given after 160s.
    2) For environment, next state will be updated asynchronously every end of phase

- Reward
    1) Max Pressure Control Theory (Reward = -pressure=-(inflow-outflow))

### Prerequisite
- python 3.7.9 above
- pytorch 1.7.1 above
- tensorboard 2.0.0 above

### How to use
check the condition state (throughput)
```shell script
    python ./run.py simulate
``` 
Run in RL algorithm DQN (default device: cpu)
```shell script
    python ./run.py train --gpu False
``` 
- check the result
Tensorboard
```shell script
    tensorboard --logdir ./training_data
``` 
Hyperparameter in json, model is in `./training_data/[time you run]/model` directory.

- replay the model
```shell script
    python ./run.py test --replay_name /replay_data in training_data dir/ --replay_epoch NUM
```

## Utils
gen_tllogic.py
```shell script
python /path/to/repo/util/gen_tllogic.py --file [xml]
```
graphcheck.py
```shell script
python /path/to/repo/util/gen_tllogic.py file_a file_b --type [edge or lane] --data speed
```
    - check the tensorboard
    `tensorboard --logdir tensorboard`
