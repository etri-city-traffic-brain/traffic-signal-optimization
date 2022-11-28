# Adaptive Traffic Signal Control ([한국어](./README.md))

Traffic signal optimization with the help of machine learning, esp., reinforcement learning.

* do multi-agent based adaptive traffic signal control
* support distributed learning
* support PPO policy
* support kc, offset, gr, gro as actions
  * kc : keep-change
  * offset : offset adjustment
  * gr :  green time adjustment
  * gro : green-time + offset adjustment
* set a level of debug message by using the DBG_OPTIONS in DebugConfiguration.py

  
<hr>

### Requirements
* [The SALT traffic simulator should be installed](https://github.com/etri-city-traffic-brain/traffic-simulator)
* Traffic simulation scenarios for the area to do traffic optimization should be prepared.
<br> <br>
* A python execution environment should be created
  * Packages to install 
    * python 3.x (version 3.8+ recommended)
    * tensorflow 2.x (version 2.3.0+ recommended)
    * keras 2.x (version 2.4.3+ recommended)
    * pandas
    * gym
    * matplotlib
    * Deprecated
  * You can refer YAML file : [uniq.opt.env.yaml](./uniq.opt.env.yaml) 
  * example
    ```shell
    ### ex1. with package installing
    conda create --name p3.8 python=3.8
    pip install tensorflow==2.3.0
    pip install keras==2.4.3
    pip install pandas
    pip install gym
    pip install matplotlib
    pip install Deprecated   

    ### ex2. with YAML file 
    #### if needed, you can change the prefix in the YAML file : should match the path (name of virtual env., user name, ...)
    #### exmaple,   prefix: /home/[USER_NAME]/Anaconda3/envs/[VIRTUAL_ENV_NAME]
    conda env create --file uniq.opt.env.yaml
   ```
 
<br> <br>
* Environment variables such as SALT_HOME and PYTHONPATH should be declared.
  * SALT_HOME : a path SALT simulator is installed
  * add library paths of SALT to PYTHONPATH
    ```shell
    example, 
      export SALT_HOME=/home/tsoexp/z.docker_test/traffic-simulator
      export PYTHONPATH="$SALT_HOME/tools:$PYTHONPATH"
      export PYTHONPATH="$SALT_HOME/tools/libsalt:$PYTHONPATH"
    ```

<hr>

### How to use ###
* run a python program run.py with several arguments
  ```shell
  usage: run.py [-h] [--mode {train,test,simulate}] [--scenario-file-path SCENARIO_FILE_PATH]
              [--map {dj_all,doan,sa_1_6_17,cdd1,cdd2,cdd3}] [--target-TL TARGET_TL]
              [--start-time START_TIME] [--end-time END_TIME] [--method {sappo}]
              [--action {kc,offset,gr,gro}] [--state {v,d,vd,vdd}]
              [--reward-func {pn,wt,wt_max,wq,wq_median,wq_min,wq_max,tt,cwq}]
              [--cumulative-training CUMULATIVE_TRAINING] [--model-num MODEL_NUM]
              [--infer-model-num INFER_MODEL_NUM] [--result-comp RESULT_COMP] [--io-home IO_HOME]
              [--epoch EPOCH] [--warmup-time WARMUP_TIME] [--model-save-period MODEL_SAVE_PERIOD]
              [--print-out PRINT_OUT] [--action-t ACTION_T]
              [--reward-info-collection-cycle REWARD_INFO_COLLECTION_CYCLE]
              [--reward-gather-unit {sa,tl,env}] [--gamma GAMMA] [--epsilon EPSILON]
              [--epsilon-min EPSILON_MIN] [--epsilon-decay EPSILON_DECAY]
              [--epoch-exploration-decay EPOCH_EXPLORATION_DECAY] [--ppo-epoch PPO_EPOCH]
              [--ppo-eps PPO_EPS] [--_lambda _LAMBDA] [--a-lr A_LR] [--c-lr C_LR]
              [--network-size NETWORK_SIZE] [--optimizer OPTIMIZER] [--actionp ACTIONP]
              [--mem-len MEM_LEN] [--mem-fr MEM_FR] [--offset-range OFFSET_RANGE]
              [--control-cycle CONTROL_CYCLE] [--add-time ADD_TIME] [--infer-TL INFER_TL]
              [--infer-model-path INFER_MODEL_PATH]
              [--num-of-optimal-model-candidate NUM_OF_OPTIMAL_MODEL_CANDIDATE]

  ```
  * see README_DIST.md if you want to do distributed learning for adaptive traffic signal control


  ####  some important arguments
    ``` 
    --mode {train,test,simulate}
      train - RL model training
      test - trained model testing
      simulate - fixed-time simulation before test

    --scenario-file-path SCENARIO_FILE_PATH
      home directory of scenario; relative path

    --map {dj_all, doan, sa_1_6_17, cdd1, cdd2, cdd3}
      name of map

    --target-TL TARGET_TL
      target signal groups; multiple groups can be separated by comma
      (ex. --target-TL SA 101,SA 104)

    --start-time START_TIME
      start time of traffic simulation; seconds

    --end-time END_TIME
      end time of traffic simulation; seconds

    --method {sappo} 
      optimizing method

    --state  {v,d,vd,vdd}
      v - volume, d - density, vd - volume + density, vdd - volume / density

    --action {kc,offset,gr,gro}
      kc - keep or change(limit phase sequence)
      offset - offset
      gr - green ratio
      gro - green ratio+offset

    --reward {pn,wt,wt_max,wq,wq_median,wq_min,wq_max,tt,cwq}
      pn - passed num, wt - wating time, wq - waiting q length, tt - travel time, 
      cwq - cumulative waiting q length
    
    --model-num MODEL_NUM
      trained model number for inference

    --result-comp True or False
      whether compare simulation result or not 

    --infer-TL INFER_TL
      signal groups to do inference with pre-trained model; 
      multiple groups can be separated by comma (ex. --infer_TL 'SA 101,SA 104')

    ```



#### to do train model
* pass 'train' as the value of the argument 'mode' when executing the program
    ```shell script
    python run.py --mode train
    
    # train SA 104 and SA 107 : control other intersecctions using fixed signal
    python run.py --mode train --method sappo --map doan  --target-TL "SA 104, SA 107"  --epoch 1 --action gro --start-time 25200 --end-time 32400 

    # train SA 1, SA 6 and SA 17 : control other intersecctions using fixed signal
    python run.py --mode train --map sa_1_6_17 --target-TL "SA 1, SA 6, SA 17" --method sappo --state vdd --action offset --reward-func cwq --epoch 1 --model-save-period 5

    # train SA 101 and SA 111  : control SA 104 and SA 107 with the inference of the trained model and control the rest of the intersections using fixed signal 
    python run.py --mode train --method sappo --target-TL "SA 101,SA 111"  --map doan  --epoch 1 --action gro --start-time 25200 --end-time 32400 --infer-TL "SA 104, SA 107"  --infer-model-num 5
    ``` 


#### to do test with learned model
* pass 'test' as the value of the argument 'mode' when executing the program
* deliver whether to perform the comparison to the fixed signal execution using the 'result-comp' argument
  * without result compare
    * only run test scenario
      * --result-comp False

      * ```shell script
      python run.py --mode test  --model-num xx --result-comp False
      python run.py --mode test  --model-num xx --result-comp False --map doan --target-TL "SA 101,SA 111"  --start-time 25200 --end-time 32400
      ``` 
  * with result compare
    * for test in **local**
      1. run fixed time scenario (ref. 'simulate' mode)
      2. run test scenario
      3. compare results
      ```shell script
      python run.py --mode test --model-num xx --result-comp True
      python run.py --mode test  --model-num xx --result-comp True --map doan --target-TL "SA 101,SA 111"  --start-time 25200 --end-time 32400
      ``` 
  
#### to do simulate with fixed-time traffic signal
* pass 'simulate' as the value of the argument 'mode' when executing the program

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
    tensorboard --logdir logs --host 192.1.2.3 --port 7007 # for remote access
    ``` 

<hr>

### Output files ###
#### Train ####
  * Total rewards for each epoch
    ```shell script
      output/train/train_epoch_total_reward.txt
    ```
  * Rewards for each epoch at each intersection
    ```shell script
      output/train/train_epoch_tl_reward.txt
    ```
  * SALT simulation output during training
    ```shell script
      output/train/_PeriodicOutput.csv
    ```
<br> <br>

#### Test ####
##### RL Scenario #####
  * phase, action, reward, statistics info(such as average speed, travel time, passed vehicle number,...) for each step at each intersection
    ```shell script
    output/test/rl_phase_reward_output.txt
    ```
  * SALT simulation output for test scenario
    ```shell script
    output/test/_PeriodicOutput.csv
    ```
<br> <br>

##### Fixed Time Scenario #####
  * phase, reward, statistics info(such as average speed, travel time, passed vehicle number,...) for each step at each intersection
    ```shell script
    output/simulate/ft_phase_output.txt
    ```
  * SALT simulation output for fixed time scenario
    ```shell script
    output/simulate/_PeriodicOutput.csv
    ```

  * When **result-comp is True**, comparison of improvement for model number xx
    ```shell script
    output/test/total_compare_output_model_num_xx.csv
    ```
