# 적응형 교통 신호 제어 ([English](./README.en.md))

강화학습에 기반한 교통 신호 최적화

* 다중 에이전트 기반 교통 신호 제어 지원
* 분산학습 지원 
* 강화학습의 정책으로 PPO 지원 
* 강화학습 수행시 행동(action)으로 kc, offset, gr, gro 지원
  * kc : 유지 혹은 변경
  * offset : 옵셋 조정
  * gr :  녹색 신호 지속 시간 조정
  * gro : 녹색 시간 + 옵셋 조정
* DBG_OPTIONS (DebugConfiguration.py) 설정을 통해 원하는 수준의 디버그 메시지 출력 설정

  
<hr>

### 적응형 교통 신호 제어를 통한 교통 신호 최적화를 위한 필요 사항
* [SALT 교통 시뮬레이터 설치] (https://github.com/etri-city-traffic-brain/traffic-simulator)
* 교통 신호 최적화를 수행할 지역에 대한 교통 시뮬레이션 시나리오 준비
<br> <br>
* 파이쎤 실행 환경 생성
  * 설치되어야 하는 패키지들 
    * python 3.x (버전 3.8 이상 추천)
    * tensorflow 2.x (버전 2.3.0 이상 추천)
    * keras 2.x (버전 2.4.3 이상 추천)
    * pandas
    * gym
    * matplotlib
    * Deprecated
  * YAML 파일 참고 : [uniq.opt.env.yaml](./uniq.opt.env.yaml) 
  
  
  * 실행 환경 생성 예
  ```shell
  ### 직접 패키지 인스톨하여 생성
  conda create --name p3.8 python=3.8
  pip install tensorflow==2.3.0
  pip install keras==2.4.3
  pip install pandas
  pip install gym
  pip install matplotlib
  pip install Deprecated   

  ### yaml 파일을 통한 생성
  #### 원하는 경우, 가상 환경의 이름을 바꾸고, prefix 를 경로에 맞게 바꾸어준다
  #### 예를 들어,   prefix: /home/[사용자이름]/Anaconda3/envs/[가상환경이름]
  conda env create --file uniq.opt.env.yaml
  ```
  
<br> <br>
* 환경변수 SALT_HOME 와 PYTHONPATH 선언
  * SALT_HOME : SALT 시뮬레이터가 설치된 디렉토리
  * PYTHONPATH에 SALT 라이브러리 경로 추가
    ```shell
    example, 
      export SALT_HOME=/home/tsoexp/z.docker_test/traffic-simulator
      export PYTHONPATH="$SALT_HOME/tools:$PYTHONPATH"
      export PYTHONPATH="$SALT_HOME/tools/libsalt:$PYTHONPATH"
    ```

<hr>

### 실행 방법 ###
* run.py라는 파이썬 프로그램 실행
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
  * 분산 학습 관련해서는 [README_DIST.md](./README_DIST.md) 참고


  ####  주요 명령행 인자 
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



#### 모델 훈련을 위한 실행 
* 프로그램을 실행할때 인자 'mode' 를 'train' 으로 설정하여 실행
    ```shell script
    example, 
      python run.py --mode train
    
      # train SA 104 and SA 107 : control other intersecctions using fixed signal
      python run.py --mode train --method sappo --map doan  --target-TL "SA 104, SA 107"  --epoch 1 --action gro --start-time 25200 --end-time 32400 

      # train SA 1, SA 6 and SA 17 : control other intersecctions using fixed signal
      python run.py --mode train --map sa_1_6_17 --target-TL "SA 1, SA 6, SA 17" --method sappo --state vdd --action offset --reward-func cwq --epoch 1 --model-save-period 5

      # train SA 101 and SA 111  : control SA 104 and SA 107 with the inference of the trained model and control the rest of the intersections using fixed signal 
      python run.py --mode train --method sappo --target-TL "SA 101,SA 111"  --map doan  --epoch 1 --action gro --start-time 25200 --end-time 32400 --infer-TL "SA 104, SA 107"  --infer-model-num 5
    ``` 
<br> <br>

#### 훈련된 모델 평가를 위한 실행
* 프로그램을 실행할때 인자 'mode' 를 'test'로 설정하여 실행
* 'result-comp' 인자를 이용하여 고정신호 수행과 결과 비교 수행 여부 전달 
  * 고정 신호 수행과 비교없이 실행 
    * 훈련된 모델을 위한 시나리오만 실행
      * --result-comp False
      ```shell script
      python run.py --mode test  --model-num xx --result-comp False
      python run.py --mode test  --model-num xx --result-comp False --map doan --target-TL "SA 101,SA 111"  --start-time 25200 --end-time 32400
      ``` 
  * 고정 신호 수행과 비교 실행 
    * 이전에 실행해 놓은 고정 신호 수행의 결과 파일과 비교
      ```shell script
      python run.py --mode test --model-num xx --result-comp True
      python run.py --mode test  --model-num xx --result-comp True --map doan --target-TL "SA 101,SA 111"  --start-time 25200 --end-time 32400
      ``` 
  
#### 고정 신호 실행
* 프로그램을 실행할 때 인자 'mode' 를 'simulate'으로 설정하여 실행
    ``` shell script
    python run.py --mode simulate  --map doan --target-TL "SA 101,SA 111"  --start-time 25200 --end-time 32400
    ```

#### Tensorboard 실행

    ```shell script
    # for local access with default port (6006)
    tensorboard --logdir logs
 
    # for remote access with default port
    tensorboard --logdir logs --host 129.2.3.4

    # for remote access with given port (7007)
    tensorboard --logdir logs --host 192.1.2.3 --port 7007 # for remote access
    ``` 

<hr>

### 결과 파일 
####  훈련(학습)
  * 각 epoch에 대한 전체 보상 
    ```shell script
      output/train/train_epoch_total_reward.txt
    ```
  * 각 epoch에 대한 교차로별 보상
    ```shell script
      output/train/train_epoch_tl_reward.txt
    ```
  * 훈련 동안 SALT 시뮬레이션의 출력
    ```shell script
      output/train/_PeriodicOutput.csv
    ```
<br> <br>

#### 평가
##### 훈련된 모델에 대한 평가
  * 각 교차로에 대한 각 스텝별 행동, 보상, 통계 정보(평균 속도, 평균 여행 시간, 통과 차량 수 등)
    ```shell script
    output/test/rl_phase_reward_output.txt
    ```
  * 훈련 모델 평가 중 SALT 시뮬레이션의 출력
    ```shell script
    output/test/_PeriodicOutput.csv
    ```
<br> <br>

##### 고정 신호 시나리오에 대한 평가
  * 각 교차로에 대한 각 스텝별 페이즈 정보, 보상, 통계 정보(평균 속도, 평균 여행 시간, 통과 차량 수 등)
    ```shell script
    output/simulate/ft_phase_output.txt
    ```
  * 고정신호 수행 중 SALT 시뮬레이션의 출력
    ```shell script
    output/simulate/_PeriodicOutput.csv
    ```

  *  **result-comp 값이 True**인 경우, 훈련된 모델에 의한 수행과 고정 신호에 의한 수행의 비교 결과 
    ```shell script
    output/test/total_compare_output_model_num_xx.csv
    ```
