### distributed learning for adaptive traffic signal control

### How to use ###
* should declare SALT_HOME and PYTHONPATH as an environment variable
    ```shell
    export SALT_HOME=/home/tsoexp/z.docker_test/traffic-simulator
    export PYTHONPATH="$SALT_HOME/tools:$PYTHONPATH"
    export PYTHONPATH="$SALT_HOME/tools/libsalt:$PYTHONPATH"
    ```
* change configurations in config.py 
* set DBG_OPTIONS in DebugConfiguration.py
* run python program as follows
  * launch controller daemon on the node responsible for controlling whole learning
    * usage
      ```shell
      usage: DistCtrlDaemon.py [-h] [--port PORT]  [--validation-criteria VALIDATION_CRITERIA]
                         [--num-of-learning-daemon NUM_OF_LEARNING_DAEMON] [--model-store-root-path MODEL_STORE_ROOT_PATH]
                         [--num-of-optimal-model-candidate NUM_OF_OPTIMAL_MODEL_CANDIDATE]
                         [--mode {train,test,simulate}] [--scenario-file-path SCENARIO_FILE_PATH]
                         [--map {dj_all,doan,sa_1_6_17}] [--target-TL TARGET_TL] [--start-time START_TIME]
                         [--end-time END_TIME] [--method {sappo}] [--action {kc,offset,gr,gro}] [--state {v,d,vd,vdd}]
                         [--reward-func {pn,wt,wt_max,wq,wq_median,wq_min,wq_max,wt_SBV,wt_SBV_max,wt_ABV,tt,cwq}]
                         [--model-num MODEL_NUM] [--result-comp RESULT_COMP] [--io-home IO_HOME] [--epoch EPOCH]
                         [--warmup-time WARMUP_TIME] [--model-save-period MODEL_SAVE_PERIOD] [--print-out PRINT_OUT]
                         [--action-t ACTION_T] [--reward-info-collection-cycle REWARD_INFO_COLLECTION_CYCLE]
                         [--reward-gather-unit {sa,tl,env}] [--gamma GAMMA] [--ppo-epoch PPO_EPOCH] [--ppo-eps PPO_EPS]
                         [--_lambda _LAMBDA] [--a-lr A_LR] [--c-lr C_LR] [--actionp ACTIONP] [--mem-len MEM_LEN]
                         [--mem-fr MEM_FR] [--offset-range OFFSET_RANGE] [--control-cycle CONTROL_CYCLE]
                         [--add-time ADD_TIME] [--infer-TL INFER_TL] [--infer-model-path INFER_MODEL_PATH]

      ```
    * some parameters
      * --validation-criteria : A performance improvement goal indicating when to stop training
      * --num-of-learning-daemon : number of learning daemon 
      * --model-store-root-path : path where trained models are stored 
      * --num-of-optimal-model-candidate : number of candidate to compare reward to find optimal model
    * example
      ```shell
      python DistCtrlDaemon.py --port 2727 --map doan  --target "SA 101, SA 104" --num-of-learning-daemon 2 --action gr --validation-criteria 5.0 --epoch 100 --model-save-period 10
      ```
    
  * launch execution daemons in every node responsible for training execution
    * usage
      ```shell
      usage: DistExecDaemon.py [-h] [--port PORT] [--ip-addr IP_ADDR]
      ```
    * parameters
      * --ip-addr : ip address to send connection request
      * --port : port to send connection request
    * example
      ```shell
      python DistExecDaemon.py --ip_addr 129.254.182.176  --port 2727
      ```
* you can run with dist_learning.sh after setup Passwordless SSH Login
  * you should some values before run shell script
    * env related parameters, (distributed) reinforcement related parameters, ...
  * How to Set Up Passwordless SSH Login
    * ref. https://phoenixnap.com/kb/setup-passwordless-ssh
