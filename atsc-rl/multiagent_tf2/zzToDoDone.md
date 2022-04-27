# -*- coding: utf-8 -*-


### todo 
* generate info to be used by visualization tool : fn_rl_phase_reward_output
* make up the code which are related infer-TL argument
* distributed traffic signal optimization
* dockerize 
* out of memory : python process terminated with "killed" message
  * TF1.x version implemented by mgpi also has same problem 

### ing
* distributed traffic signal optimization
  * make up code, esp., command generation 

### done history
* Tag v1.0-20220426PM-dev-tf2
  * multiple SA train/test succ
  * separate action/reward mgmt related code into independent class from env
  * multiple model(trained separately) load & test succ
  * single SA train/test succ
  * implement env : sappo
  * implement PPO with TF 2.x

