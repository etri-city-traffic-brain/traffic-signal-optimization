

python run.py --mode test --map doan --target "SA 101,SA 104"  --action offset --model-num 0
python run.py --mode test --map doan --target "SA 101,SA 104"  --action gr --model-num 0
python run.py --mode train --map doan --target "SA 101,SA 104" --action gr --epoch 1 --model-num 0 --reward-func cwq
python run.py --mode train --map doan --target "SA 101,SA 104" --action gr --epoch 1 --model-num 0 --reward-func pn

python run.py --mode simulate --map doan --target "SA 101" --action gr --epoch 1 --model-num 0 --reward-func pn
python run.py --mode train --map doan --target "SA 101" --action gr --epoch 1 --model-num 0 --reward-func pn
python run.py --mode test --map doan --target "SA 101" --action gr --epoch 1 --model-num 0 --reward-func pn --model-num 0 --result-comp true



### todo
* experiments
  * increase dimension : [2048, 1024, 512, 256, 128]
  * change control_cycle
  
* group split
  * make code which can work with small node
    * can work when the # of node is less than the # of target (i.e., # of node < # of target)
* [d] Fix "ModuleNotFoundError: No module named "libsalt" " "
  * this happens sometimes(not always) when we run with shell script(dist_learning.sh)
  
* [d] fix error in compareResult() at ResultCompare.py
  * start time of comparison : should consider argument and scenario file

* optimal model num
  * keep record while whole distributed learning 
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

* shell script
  * add nohup into script : nohup .... > foo.out 2>&1 &
    ```shell
     nohup python DistCtrlDaemon.py --port 2828 --map sa_1_6_17 --target "SA 6" --num-of-learning-daemon 1 \
                                   --state vdd  --action gro --reward-func cwq --validation-criteria 10.0 \
                                   --epoch 200 --model-save-period 5 --cumulative-training True \
                                   --model-store-root-path /home/tsoexp/share/results/2828/gro_cwq > ctrl.gro.cwq 2>&1 &
    nohup python DistExecDaemon.py --ip-addr 101.79.1.116  --port 2828 > exec.gro.cwq 2>&1 &
    nohup tensorboard --logdir ./logs --host 101.79.1.116 --port 7007 > tb.out 2>&1 &
    ```

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
    
* distributed traffic signal optimization
  * make LearningDaemonThread::__copyTrainedModel() work with various method
    * currently only care sappo

* solve questions :  todos in the code
  * can find given questions in the code
    * grep todo *
    
* dockerize

<hr>
  
### done history

* Tag v1.5.1-
  * add cumulative trainint ,  --cumulative-training 
    * Cumulative training based on a previously trained model parameter
    
* Tag v1.5-20220519AM-dist
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

