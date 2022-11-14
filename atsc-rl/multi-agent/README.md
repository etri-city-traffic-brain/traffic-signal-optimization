# ATSC-RL(Multi-Agent)

<br>
<br>
<br>

### This version of the code is too old.
### There are no plans to maintain(currently not supported).

<br>
<br>
<br>

<hr>



### How to use ###
- change libsalt directory in config.py

train mode
```shell script
    python run.py --mode train
``` 
test mode(without result compare)
- only run test scenario
- run fixed time and test scenario separately at the same time on the **Modutech system**
```shell script
    python run.py --mode test --model-num xx
``` 
test mode(with result compare)
- for test in **local**
1. run fixed time scenario
2. run test scenario
3. compare results
```shell script
    python run.py --mode test --model-num xx --resultComp True
``` 
Tensorboard
```shell script
    tensorboard --logdir logs # for local access
    tensorboard --logdir logs --host 0.0.0.0 # for remote access
``` 

### Output ###
#### Train ####
- Total rewards for each epoch
```shell script
    output/train/train_epoch_total_reward.txt
```
- Rewards for each epoch at each intersection
```shell script
    output/train/train_epoch_tl_reward.txt
```
- SALT simulation output during training
```shell script
    output/train/-PeriodicOutput.csv
```

#### Test ####
##### RL Scenario #####
- Actions and rewards for each step at each intersection
```shell script
    output/test/rl_phase_reward_output.txt
```
- SALT simulation output for test scenario
```shell script
    output/test/-PeriodicOutput.csv
```
##### Fixed Time Scenario #####
- phase for each step at each intersection
```shell script
    output/simulate/ft_phase_output.txt
```
- SALT simulation output for fixed time scenario
```shell script
    output/simulate/-PeriodicOutput.csv
```

- When **resultComp is True**, comparison of improvement for model number {}
```shell script
    output/test/total_compare_output_model_num_{}.csv
```
