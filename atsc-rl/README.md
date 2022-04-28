### 인공지능 기반의 적응형 교통 신호 제어 모델 개발
**A**daptive **T**raffic **S**ignal **C**ontrol with **R**einforcement **L**earning

#### multiagent_tf2
* do multi-agent based adaptive traffic signal control
* support distributed learning
  1. divide the target SAs
  2. allocate SA to be in charge of learning to multiple nodes
  3. each node learns about the SA responsible 
  4. gather learned results : trained model parameters
  5. do experiment with trained models to see performance improvement
  6. repeat above steps until performance improvement goal is achieved
  
* support policy
  * ppo
* supported actions
  * kc : keep-change
  * offset : offset adjustment
  * gr : green-ratio adjustment
  * gro : green-ratio + offset adjustment
* one env exists : SappoEnv
  * considered multiple SA support
* separate action/reward related code from env
  * make an independent class for action mgmt
  * make an independent class for reward mgmt
  
#### multi-agent : work on TF 1.x
* do multi-agent based adaptive traffic signal control
* support policy
  * ddqn, ppo
* supported actions
  * kc : keep-change
  * offset : offset adjustment
  * gr : green-ratio adjustment
  * gro : green-ratio + offset adjustment
* env exists per action
  * envs
    * SALT_doan_multi_PSA : ddqn + kc
    * SALT_SAPPO_noConst : ppo + kc
    * SALT_SAPPO_offset : ppo + offset (multiple SA)
    * SALT_SAPPO_offset_single : ppo + offset (single SA)
    * SALT_SAPPO_offset_EA : ppo + ?
    * SALT_SAPPO_green_single : ppo + gr (single SA) 
    * SALT_SAPPO_green_offset_single : ppo + gro (single SA)
  * only some env considered multiple SA support
  
#### single agent