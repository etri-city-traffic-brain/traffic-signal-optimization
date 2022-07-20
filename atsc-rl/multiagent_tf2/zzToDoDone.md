### todo
* change validation-criteria into target-improvement-rate
  * change
    * README_DIST.md, dist.run.single.sh, dist_learning.sh
    * DistCtrlDaemon.py
  * remove file
    * run.single.sh
    
* result compare
  * use scenario id
  
* experiments
  * increase dimension : [2048, 1024, 512, 256, 128]
  * change control_cycle

* optimal model num
  * [d] keep record while whole distributed learning : KEEP_OPTIMAL_MODEL_NUM
    * use append(), readlines()[-1] 
  * enlarge the scope of candidate
    * consider control_cycle
  * adjust the starting point of scanning the candidate : after 1/2 point

* state
  * change to use not only zero-hop info but also 1-hop info
    * currently we use only zero-hop inf`o
    * use_zero_hop_only == True or False`
  * include action of adjacent TL(SA) in the state info
    * I wonder
      * it is possible?
      * it has meaning?

* make up the code
  * [d] infer-TL argument related code
  * remove "if 0:" related codes
  * debugging related code 
    * Print*, RunWithWaitForDebug, RunWithDistributed
  * constants
    * [d] sim_period, state_weight, reward_weight in SappoEnv.py

* [ing] out of memory problem : python process terminated with "killed" message 
  * TF1.x version implemented by mgpi also has same problem
  * [d] memory leaks in simulator
    * check what happens when we do repeat simulate mode
    * also has same problem : size of memory used by process increase
    * make deallocate memory when simulation is terminated : by hwsong
  * should deallocate memory
    * delete ndarray in PPOAgentTF2::replay() --> replayWithDel()
      * As # of replay memory entry grows, so does the incrememt of memory  재현 메모리 상의 엔트리 수가 커짐에 따라 메모리 증가도 커진다.
      * adjust maximum replay memory size
    * 최대 개수 초과시 엔트리 삭제하는 경우 확인 ... 메모리 반납
  * use memory_profiler
    * ref. https://pypi.org/project/memory-profiler/
    * ref. https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
    
    


* solve questions :  todos in the code
  * can find given questions in the code
    * grep todo *
    
* distributed traffic signal optimization
  * make LearningDaemonThread::__copyTrainedModel() work with various method
    * currently only care sappo
    * 

<hr>
  

### done history
* Tag V1.3a-202208
  * add some statistical info into TSO output file for viz tool
    * sum travel time, num of passed vehicle, ... 
* Tag V1.2a-20220719
  * (addition) add some func.s into shell script for distributed trainging such as simulate, train, terminate, clean, ...
    * Rename dist_learning.sh to dist_training.sh
* Tag V1.2a-20220706
  * (removal) exclude wt_SBV, wt_SBV_max and wt_ABV from reward functions because it is TOO SLOW
  
  * use command line arguments instead of configuration file
    * network-size, optimizer
  
  * (addition) dump average travel time of intersection into output file of optimizer(ft/rl_phase_reward_output.txt) for VizTool
    * DBG_OPTINS.WithAverageTravelTime
    * output file
      * before : step,tl_name,actions,phase,reward,avg_speed
      * after : step,tl_name,actions,phase,reward,avg_speed,avg_travel_time
      
  * (stablization) Fixed the problem that boost lib could not be found when performing reinforcement learning
    * pass not some but all environment variables of the user when we launch RL program
      * before : some env variables of the user
      * now : all env variables of the user

  * (improvement) compile salt in the docker image env
    * from : make a binary outside and copy it to docker image env
    * to : copy SALT source into docker image env and compile it to make binary
    
  * (improvement) fix and extend result compare related stuff
    * fix : logic to calculate the improvement ratio of travel time
    * extend : calculate improvement rate for each SA
  
* Tag v1.1a-20220602
  * add cumulative training ,  --cumulative-training 
    * Cumulative training based on a previously trained model parameter
  * add codes to control exploration ratio when we do train : USE_EXPLORATION_EPSILON
  * group split
    * make code which can work with small node
      * can work when the # of node is less than the # of target (i.e., # of node < # of target)
  * Fix "ModuleNotFoundError: No module named "libsalt" " "
    * this happens sometimes(not always) when we run with shell script(dist_learning.sh)
  * fix error in compareResult() at ResultCompare.py
    * start time of comparison : should consider argument and scenario file
  * add nohup into script : nohup python foo.py .... > foo.out 2>&1 &

    
* Tag v1.0.5-20220519AM-dist
  * distributed traffic signal optimization
    * command generation : generateCommand() at TSOUtil.py
      * considered all arguments used to run a single node program(ref. parseArgument() at run.py)
    * finding optimal model num : findOptimalModelNum() at TSOUtil.py
      * considered the case that epoch is very small 
    * check syntax of assert stmt
      * syntax : assert [condition], [error msg]
  * add operation _REWARD_GATHER_UNIT_
    * gather reward related info per TL
    * calculate by _REWARD_GATHER_UNIT_
      * calculateRewardByUnit(sa_idx, unit) returns calculated rewards
  * generate info to be used by visualization tool : fn_rl_phase_reward_output
    * appendPhaseRewards() at SappoEnvUtil.py
      * called in SaltSappoEnvV3::step(), SaltSappoEnvV3::reset() at SappoEnv.py
    * fixedTimeSimulate() at run.py

* Tag v1.0-20220426PM-dev-tf2
  * multiple SA train/test succ
  * separate action/reward mgmt related code into independent class from env
  * multiple model(trained separately) load & test succ
  * single SA train/test succ
  * implement env : sappo
  * implement PPO with TF 2.x



### dummy.....
python run.py --mode test --map doan --target-TL "SA 101,SA 104"  --action offset --model-num 0
python run.py --mode test --map doan --target-TL "SA 101,SA 104"  --action gr --model-num 0
python run.py --mode train --map doan --target-TL "SA 101,SA 104" --action gr --epoch 1 --model-num 0 --reward-func cwq
python run.py --mode train --map doan --target-TL "SA 101,SA 104" --action gr --epoch 1 --model-num 0 --reward-func pn

python run.py --mode simulate --map doan --target-TL "SA 101" --action gr --epoch 1 --model-num 0 --reward-func pn
python run.py --mode train --map doan --target-TL "SA 101" --action gr --epoch 1 --model-num 0 --reward-func pn
python run.py --mode test --map doan --target-TL "SA 101" --action gr --epoch 1 --model-num 0 --reward-func pn \
              --model-num 0 --result-comp true


python DistCtrlDaemon.py --port 2727 --num-of-learning-daemon 1 --validation-criteria 6.0 \
          --model-store-root-path /home/tsoexp/share/dl_test_1 --num-of-optimal-model-candidate 3 --scenario-file-path data/envs/salt \
          --map doan --target-TL 'SA 101, SA 104' --method sappo --state vdd --action gr --reward-func pn --model-save-period 1 --epoch 5

python DistCtrlDaemon.py --port 2727 --num-of-learning-daemon 2 --validation-criteria 10.0 \
          --model-store-root-path /home/tsoexp/share/dl_test_1 --num-of-optimal-model-candidate 3 --scenario-file-path data/envs/salt \
          --map doan --target-TL 'SA 101, SA 104, SA 111' --method sappo --state vdd --action gr --reward-func pn --model-save-period 1 \
          --cumulative-training true --epoch 3


python DistExecDaemon.py --ip-addr 129.254.182.176 --port 2727
