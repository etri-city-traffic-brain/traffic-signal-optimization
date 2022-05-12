

python run.py --mode test --map doan --target "SA 101,SA 104"  --action offset --model-num 0

python run.py --mode test --map doan --target "SA 101,SA 104"  --action gr --model-num 0
python run.py --mode train --map doan --target "SA 101,SA 104" --action gr --epoch 1 --model-num 0 --reward-func cwq
python run.py --mode train --map doan --target "SA 101,SA 104" --action gr --epoch 1 --model-num 0 --reward-func pn




### todo
* distributed traffic signal optimization
  * make LearningDaemonThread::__copyTrainedModel() work with various method
    * currently only care sappo
* make up the code
  * infer-TL argument related code
  * debugging related code 
    * Print*, RunWithWaitForDebug, RunWithDistributed
  * [d] constants
    * sim_period, state_weight, reward_weight in SappoEnv.py
* [d] add operation _REWARD_GATHER_UNIT_
  * gather related info per TL
  * calculate by _REWARD_GATHER_UNIT_
    * calculateRewardByUnit(sa_idx, unit) returns calculated rewards
    
* [d] generate info to be used by visualization tool : fn_rl_phase_reward_output
  * SaltSappoEnvV3::__appendPhaseRewards() at SappoEnv.py
    * called in SaltSappoEnvV3::step(), SaltSappoEnvV3::reset() at SappoEnv.py
    * **보상 수집을 교차로 별로 하도록**
  * fixedTimeSimulate() at run.py

* [ing] out of memory problem
  * should deallocate memory
    * delete ndarray in PPOAgentTF2::replay() --> replayWithDel()
      * As # of replay memory entry grows, so does the incrememt of memory  재현 메모리 상의 엔트리 수가 커짐에 따라 메모리 증가도 커진다.
      * adjust maximum replay memory size
    * 최대 개수 초과시 엔트리 삭제하는 경우 확인 ... 메모리 반납
  * use memory_profiler
    * ref. https://pypi.org/project/memory-profiler/
    * ref. https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
    
* change to use not only zero-hop info but also 1-hop info
  * currently we use only zero-hop info
  * use_zero_hop_only == True or False
  
* make experimental env in the cloud(PurpleStones)
  * 3 nodes and shared storage

* solve questions :  todos in the code
  * can find given questions in the code
    * grep todo *
    
* dockerize 

* out of memory : python process terminated with "killed" message
  * TF1.x version implemented by mgpi also has same problem
  * **check what happens when we do simulate mode**
    * if memory usage increase much slowly 
      * can be confident that it is only optimizer's problem  
      * no problems in the simulator 

* cumulative learning(?)
  * learn after loading the previously learned model parameter in distributed learning

<hr>
  
### done history
* Tag v1.0-2022
  * make up the code 
    * the location of import stmt 
  * distributed traffic signal optimization
    * command generation : generateCommand() at TSOUtil.py
      * considered all arguments used to run a single node program(ref. parseArgument() at run.py)
    * finding optimal model num : findOptimalModelNum() at TSOUtil.py
      * considered the case that epoch is very small 
    * check syntax of assert stmt
      * syntax : assert [condition], [error msg]
      * 
* Tag v1.0-20220426PM-dev-tf2
  * multiple SA train/test succ
  * separate action/reward mgmt related code into independent class from env
  * multiple model(trained separately) load & test succ
  * single SA train/test succ
  * implement env : sappo
  * implement PPO with TF 2.x

