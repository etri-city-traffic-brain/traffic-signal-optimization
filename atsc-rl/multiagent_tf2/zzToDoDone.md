### todo 

* [ing] distributed traffic signal optimization
  * check code
    * [d] command generation : generateCommand() at TSOUtil.py
    * [i] find optimal model num : findOptimalModelNum() at TSOUtil.py
      * when epoch is very small 
  * make LearningDaemonThread::__copyTrainedModel() work with various method
    * currently only care sappo
* [ing] make up the code
  * infer-TL argument related code
  * [d] import stmt
  * debugging related code 
    * Print*, RunWithWaitForDebug, RunWithDistributed

* generate info to be used by visualization tool : fn_rl_phase_reward_output

* make experimental env in the cloud(PurpleStones)
  * 3 nodes and shared storage

* solve questions :  todos in code
  * find given questions in code
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
    * ...
* Tag v1.0-20220426PM-dev-tf2
  * multiple SA train/test succ
  * separate action/reward mgmt related code into independent class from env
  * multiple model(trained separately) load & test succ
  * single SA train/test succ
  * implement env : sappo
  * implement PPO with TF 2.x

