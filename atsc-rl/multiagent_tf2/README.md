# ATSC-RL(MultiAgent TF2)

### How to use ###
- change libsalt directory in config.py



#### train mode
    ```shell script
    python run.py --mode train
    
    # train SA 104 and SA 107 : control other intersecctions using fixed signal
    python run.py --mode train --method sappo --target-TL "SA 104,SA 107"  --map doan  --epoch 1 --action gro --start-time 25200 --end-time 32400 
    
    # train SA 101 and SA 111  : control SA 104 and SA 107 with the inference of the trained model and control the rest of the intersections using fixed signal 
    python run.py --mode train --method sappo --target-TL "SA 101,SA 111"  --map doan  --epoch 1 --action gro --start-time 25200 --end-time 32400 --infer-TL "SA 104, SA 107"  --model-num 0
    ``` 


#### test mode
* without result compare
  - only run test scenario
  - run fixed time and test scenario separately at the same time on the **Modutech system**
    ```shell script
    python run.py --mode test  --model-num xx --result-comp False
    python run.py --mode test  --model-num xx --result-comp False --map doan --target-TL "SA 101,SA 111"  --start-time 25200 --end-time 32400
    ``` 
* with result compare
  - for test in **local**
    1. run fixed time scenario
    2. run test scenario
    3. compare results
    ```shell script
    python run.py --mode test --model-num xx --result-comp True
    python run.py --mode test  --model-num xx --result-comp True --map doan --target-TL "SA 101,SA 111"  --start-time 25200 --end-time 32400

    ``` 
  
#### simulate mode
    ``` shell script
    python run.py --mode simulate  --map doan --target-TL "SA 101,SA 111"  --start-time 25200 --end-time 32400
    ```

#### Tensorboard
    ```shell script
    # for local access with default port (6006)
    tensorboard --logdir logs
 
    # for remote access with default port
    tensorboard --logdir logs --host 129.2.3.4

    # for remote access with given port (7007)
    tensorboard --logdir logs --host 129.2.3.4 --port 7007 # for remote access
    ``` 

<hr>

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

- When **result-comp is True**, comparison of improvement for model number {}
```shell script
    output/test/total_compare_output_model_num_{}.csv
```
