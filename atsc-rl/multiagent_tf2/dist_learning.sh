#!/bin/bash
#

# 0. set parameters
if [ 1 ]; then
  ## control
  DO_SIMULATE=false # whether do simulation with fixed signal to get ground zero performance or not
  DO_EVAL=true # whether do execution or not;  do evaluate command if true, otherwise just dump commands

  ## env related parameters
  ACCOUNT="tsoexp"
  CTRL_DAEMON_IP="129.254.182.176"
  EXEC_DAEMON_IPS=(  "129.254.182.176"  )
#  EXEC_DAEMON_IPS=(  "129.254.182.176"  "129.254.184.53"  "129.254.182.176" )
  PORT=2727
  EXEC_DIR=/home/tsoexp/PycharmProjects/traffic-signal-optimization-for-dist/atsc-rl/multiagent_tf2

#  CTRL_DAEMON_IP="101.79.1.126" # "129.254.182.176"
#  EXEC_DAEMON_IPS=(  "101.79.1.112"   "101.79.1.115" )
#  #EXEC_DAEMON_IPS=(  "101.79.1.112"   "101.79.1.115"  "101.79.1.116"  "101.79.1.126"  )
#  PORT=2727
#  EXEC_DIR=/home/tsoexp/z.uniq/traffic-signal-optimization-for-dist/atsc-rl/multiagent_tf2/

  ACTIVATE_CONDA_ENV_P3_8="source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate p3.8 "

  CTRL_DAEMON="DistCtrlDaemon.py"
  EXEC_DAEMON="DistExecDaemon.py"
  RL_PROG="run.py"

  ## output file : to save verbosely dumped messages
  POST_OUT=`date +"%F-%H-%M-%S"`
  CTRL_OUT="zz.out.ctrl.$POST_OUT"
  EXEC_OUT="zz.out.exec.$POST_OUT"

  ## Reinforcement Learning related parameters
  RL_SCENARIO_FILE_PATH="data/envs/salt"
  RL_MAP="doan"
  RL_TARGET="SA 101" #"SA 101, SA 104"
  RL_METHOD="sappo"
  RL_STATE="vdd" # v, d, vd, vdd
  RL_ACTION="gr"  # offset, gr, gro, kc
  RL_REWARD="pn"  # wq, cwq, pn, wt, tt
  RL_EPOCH=5
  RL_MODEL_SAVE_PERIOD=1


  ## distributed Reinforcement Learning related parameters
  VALIDATION_CRITERIA=6.0
  NUM_EXEC_DAEMON=${#EXEC_DAEMON_IPS[@]}
  MODEL_STORE_ROOT_PATH="/home/tsoexp/share/dl_test_1"
  NUM_OF_OPTIMAL_MODEL_CANDIDATE=3


  ## set libsalt path
  SALT_HOME=/home/tsoexp/z.docker_test/traffic-simulator
#  PYTHONPATH="$SALT_HOME/tools:$PYTHONPATH"
#  PYTHONPATH="$SALT_HOME/tools/libsalt:$PYTHONPATH"

fi



# 1. get the performance before doing reinforcement learning as a base perdormance
if $DO_SIMULATE
then


  ## 1.1 construct command
  INNER_CMD="SALT_HOME=$SALT_HOME python $RL_PROG --mode simulate --scenario-file-path $RL_SCENARIO_FILE_PATH "
  INNER_CMD="$INNER_CMD --map $RL_MAP --target-TL '$RL_TARGET' --method $RL_METHOD "
  INNER_CMD="$INNER_CMD --state $RL_STATE --action $RL_ACTION --reward-func $RL_REWARD "

  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV_P3_8; "
  CMD="$CMD cd $EXEC_DIR; "
  CMD="$CMD $INNER_CMD \" "
  echo [%] $CMD

  ## 1.2 evaluate command
  if $DO_EVAL
  then
    eval $CMD
  fi

  #  python run.py --mode simulate --map $RL_MAP --target-TL $RL_TARGET --method $RL_METHOD --state $RL_STATE  \
  #         --action $RL_ACTION --reward-func $RL_REWARD
fi


# 2. execute controller daemon
if [ 1 ]; then
  ## 2.1 construct command
  INNER_CMD="SALT_HOME=$SALT_HOME python $CTRL_DAEMON --port $PORT --num-of-learning-daemon $NUM_EXEC_DAEMON "
  INNER_CMD="$INNER_CMD --validation-criteria $VALIDATION_CRITERIA "
  INNER_CMD="$INNER_CMD --model-store-root-path $MODEL_STORE_ROOT_PATH "
  INNER_CMD="$INNER_CMD --num-of-optimal-model-candidate $NUM_OF_OPTIMAL_MODEL_CANDIDATE "

  INNER_CMD="$INNER_CMD --scenario-file-path $RL_SCENARIO_FILE_PATH "
  INNER_CMD="$INNER_CMD--map $RL_MAP --target-TL '$RL_TARGET' --method $RL_METHOD "
  INNER_CMD="$INNER_CMD --state $RL_STATE --action $RL_ACTION --reward-func $RL_REWARD "
  INNER_CMD="$INNER_CMD --model-save-period $RL_MODEL_SAVE_PERIOD --epoch $RL_EPOCH "

  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV_P3_8; "
  CMD="$CMD cd $EXEC_DIR; "
  # CMD="$CMD $INNER_CMD  \" &"
  CMD="$CMD $INNER_CMD > $CTRL_OUT 2>&1 & \" &"

  echo [%] $CMD

  ## 2.2 evaluate command
  if $DO_EVAL
  then
    eval $CMD
  fi
fi

sleep 5



# 3. launch execution daemon
for ip in ${EXEC_DAEMON_IPS[@]}
do
  ## 3.1 construct command
  INNER_CMD="SALT_HOME=$SALT_HOME python $EXEC_DAEMON --ip-addr $CTRL_DAEMON_IP --port $PORT "

  CMD="ssh $ACCOUNT@$ip  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV_P3_8; "
  CMD="$CMD cd $EXEC_DIR; "
  # CMD="$CMD $INNER_CMD \" &"
  CMD="$CMD $INNER_CMD > $EXEC_OUT 2>&1 & \" &"

  echo [%] $CMD

  ## 2.2 evaluate command
  if $DO_EVAL
  then
    eval $CMD
  fi

done

#!/bin/bash
#echo "1st param = $1"
#echo "2nd param = $2"
#echo "The number of params = $#"
#echo "all param : $@"
#
#DEFAULT_CTRL_DAEMON_IP="129.254.182.176"
#CTRL_DAEMON_IP=${1:-$DEFAULT_CTRL_DAEMON_IP}


#if [ 1 --eq 0 ]; then
  #-- following is work well
  # echo ssh $ACCOUNT@$ip  "source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate p3.8; cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT"
  # ssh $ACCOUNT@$ip  "source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate p3.8; cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" &
  ##

  #-- following is nor work well
  # echo  ssh $ACCOUNT@$ip "cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT"
  # ssh $ACCOUNT@$ip "cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" &   # error
  ##
  # echo ssh $ACCOUNT@$ip ". ~/.bashrc; cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" & # error
  # ssh $ACCOUNT@$ip ". ~/.bashrc; cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" & # error


  #python DistCtrlDaemon.py --port 2727 --map doan --target "SA 101" --num-of-learning-daemon 1 --validation-criteria 5.0
  #python DistCtrlDaemon.py --port 2727 --map doan --target "SA 101, SA 104" --num-of-learning-daemon 2 --validation-criteria 5.0
  #python DistCtrlDaemon.py --port 2727 --map doan  --target "SA 101, SA 104" --num-of-learning-daemon 2 --action gr --validation-criteria 5.0 --epoch 1 --model-save-period 1
  #
  #python DistExecDaemon.py --ip-addr 129.254.182.176  --port 2727

  # python run.py --mode train --map doan --target "SA 101,SA 104" --action offset --epoch
# fi
