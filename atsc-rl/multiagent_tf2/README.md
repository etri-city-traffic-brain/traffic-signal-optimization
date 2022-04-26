# ATSC-RL(MultiAgent TF2)

### How to use ###
* change libsalt directory in config.py
* run python program run.py with several arguments


####  some important arguments
    ``` 
    --mode {train,test,simulate}
      train - RL model training, test - trained model testing, simulate - fixed-time simulation before test

    --scenario-file-path SCENARIO_FILE_PATH
      home directory of scenario; relative path
    --map {dj_all,doan,doan_20211207,sa_1_6_17}
      name of map
    --target-TL TARGET_TL
      target signal groups; multiple groups can be separated by comma(ex. --target-TL SA 101,SA 104)
    --start-time START_TIME
      start time of traffic simulation; seconds
    --end-time END_TIME
      end time of traffic simulation; seconds

    --method {sappo} 
      optimizing method
    --state  {v,d,vd,vdd}
      v - volume, d - density, vd - volume + density, vdd - volume / density
    --action {kc,offset,gr,gro}
      kc - keep or change(limit phase sequence), offset - offset, gr - green ratio, gro - green ratio+offset
    --reward {pn,wt,wt_max,wq,wq_median,wq_min,wq_max,wt_SBV,wt_SBV_max,wt_ABV,tt,cwq}
      pn - passed num, wt - wating time, wq - waiting q length, tt - travel time, cwq - cumulative waiting q length, SBV - sum-based, ABV - average-based
    
    --model-num MODEL_NUM
      trained model number for inference
    --result-comp True or False
      whether compare simulation result or not 
    ```

* you can find full arguments and more detailed sescription by typing "python run.py -h"



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
