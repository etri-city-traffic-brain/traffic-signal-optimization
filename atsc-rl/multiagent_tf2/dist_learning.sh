#!/bin/bash
#
##
## script control
##
DO_SIMULATE=false # whether do simulation with fixed signal to get ground zero performance or not
DO_EVAL=true # whether do execution or not;  do evaluate commands if true, otherwise just dump commands

# 0. set parameters
if [ 1 ]; then
  #######
  ## env related parameters
  ##

  ###--- account(user id)
  ACCOUNT="tsoexp"
  #

  ###--- ip address of node to run control daemon
  CTRL_DAEMON_IP="129.254.184.123"  # 101.79.1.126

  ###--- ip address of nodes to run execution daemon
  EXEC_DAEMON_IPS=( "129.254.184.123" 
                    "129.254.184.184"
                    "129.254.184.238"
                    "129.254.184.248")
  #
  # 129.254.184.239		uniq1
  # 129.254.184.241		uniq2
  # 129.254.184.123		uniq3
  # 129.254.184.184		uniq4
  # 129.254.184.238		uniq5
  # 129.254.184.248		uniq6
  # 129.254.184.53		uniq7
  # 129.254.184.54		uniq8

  ###--- number of execution daemon
  NUM_EXEC_DAEMON=${#EXEC_DAEMON_IPS[@]}

  ###--- port to communicate btn ctrl daemon and exec daemon
  PORT=2727 #2727 3001  3101  3201  3301

  ###--- port for tensorboard 
  TB_PORT=6006 #6006 7001 7101 7201 7301

  ###--- directory for traffic signal optimization(TSO) execution
  EXEC_DIR=/home/tsoexp/z.uniq/traffic-signal-optimization/atsc-rl/multiagent_tf2

  ###--- conda environment for TSO
  CONDA_ENV_NAME="UniqOpt.p3.8"
  ACTIVATE_CONDA_ENV="source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate $CONDA_ENV_NAME "

  ###-- libsalt path
  SALT_HOME=/home/tsoexp/z.uniq/traffic-simulator

  #######
  ## exec program
  ##
  ###--- control daemon for distributed training
  CTRL_DAEMON="DistCtrlDaemon.py"

  ###--- execution daemon for distributed training
  EXEC_DAEMON="DistExecDaemon.py"

  ###-- reinforcement learning main
  RL_PROG="run.py"

  #######
  ## output file : to save verbosely dumped messages
  ##
  ###--- postfix of file name
  FN_POSTFIX=`date +"%F-%H-%M-%S"`

  ###--- for control daemon
  FN_CTRL_OUT="zz.out.ctrl.$FN_POSTFIX"

  ###--- for execution daemon
  FN_EXEC_OUT="zz.out.exec.$FN_POSTFIX"

  ###--- for tensorboard
  FN_TB_OUT="zz.out.tb.$FN_POSTFIX"


  #######
  ## Reinforcement Learning related parameters
  ##
  ###--- to access simulation scenario file(relative path)
  RL_SCENARIO_FILE_PATH="data/envs/salt"

  ###--- name of map to simulate
  RL_MAP="doan"

  ###--- target to train
  RL_TARGET="SA 101, SA 104, SA 107, SA 111" # SA 101,SA 104,SA 107,SA 111"

  ###--- RL method
  RL_METHOD="sappo"

  ###--- state, action, reward for RL
  RL_STATE="vdd" # v, d, vd, vdd
  RL_ACTION="gr"  # offset, gr, gro, kc
  RL_REWARD="wq"  # wq, cwq, pn, wt, tt

  ###--- training epoch
  RL_EPOCH=5	# 200

  ###--- interval for model saving : how open save model
  RL_MODEL_SAVE_PERIOD=1


  #######
  ## distributed Reinforcement Learning related parameters
  ##
  ###--- training improvement goal
  IMPROVEMENT_GOAL=20.0

  ###-- shared directory;
  ###-- should be accessed by all ctrl/exec daemon
  MODEL_STORE_ROOT_PATH="/home/tsoexp/share/dist_training"

  ###--- directory to save training result
  TODAY=`date +"%g%m%d"`  # 220701
  EXP_OPTION="rm"
  RESULT_DIR=${TODAY}/${RL_ACTION}_${RL_REWARD}_${EXP_OPTION} # 220701/gr_wq_rm


  ###--- number of optimal model candidate
  NUM_OF_OPTIMAL_MODEL_CANDIDATE=10

  ###--- whether do copy simulation output file or not : PeriodicOutput, rl_phase_reward_output,
  COPY_SIMULATION_OUTPUT="yes" # yes, true, t, TRUE, ... no, False, f

  ###-- whether do cumulative training or not : model, replay memory
  CUMULATIVE_TRAINING="True"

fi



# 1. get the performance before doing reinforcement learning as a base perdormance
if $DO_SIMULATE
then


  ## 1.1 construct command
  INNER_CMD="SALT_HOME=$SALT_HOME nohup python $RL_PROG --mode simulate --scenario-file-path $RL_SCENARIO_FILE_PATH "
  INNER_CMD="$INNER_CMD --map $RL_MAP --target-TL '$RL_TARGET' --method $RL_METHOD "
  INNER_CMD="$INNER_CMD --state $RL_STATE --action $RL_ACTION --reward-func $RL_REWARD "

  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
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
  echo
  echo
  echo "Simulation with fixed signal to get ground zero performance was done."
  echo "So base performance with dixed signal was gathered."
  echo "Now... do set DO_SIMULATE false, and launch this shell script to do distributed training"
  exit
fi


# 2. execute controller daemon
if [ 1 ]; then
  ## 2.1 construct command
  INNER_CMD="SALT_HOME=$SALT_HOME nohup python $CTRL_DAEMON --port $PORT --num-of-learning-daemon $NUM_EXEC_DAEMON "
  INNER_CMD="$INNER_CMD --validation-criteria $IMPROVEMENT_GOAL "
  INNER_CMD="$INNER_CMD --num-of-optimal-model-candidate $NUM_OF_OPTIMAL_MODEL_CANDIDATE "
  INNER_CMD="$INNER_CMD --cumulative-training $CUMULATIVE_TRAINING "
	INNER_CMD="$INNER_CMD --model-store-root-path $MODEL_STORE_ROOT_PATH/$RESULT_DIR "
  INNER_CMD="$INNER_CMD --copy-simulation-output $COPY_SIMULATION_OUTPUT "

  INNER_CMD="$INNER_CMD --scenario-file-path $RL_SCENARIO_FILE_PATH "
  INNER_CMD="$INNER_CMD--map $RL_MAP --target-TL '$RL_TARGET' --method $RL_METHOD "
  INNER_CMD="$INNER_CMD --state $RL_STATE --action $RL_ACTION --reward-func $RL_REWARD "
  INNER_CMD="$INNER_CMD --model-save-period $RL_MODEL_SAVE_PERIOD --epoch $RL_EPOCH "


  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
  CMD="$CMD cd $EXEC_DIR; "
  # CMD="$CMD $INNER_CMD  \" &"
  CMD="$CMD $INNER_CMD > $FN_CTRL_OUT 2>&1 & \" &"

  echo
  echo [%] $CMD
  echo

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
  INNER_CMD="SALT_HOME=$SALT_HOME nohup python $EXEC_DAEMON --ip-addr $CTRL_DAEMON_IP --port $PORT "

  CMD="ssh $ACCOUNT@$ip  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
  CMD="$CMD cd $EXEC_DIR; "
  # CMD="$CMD $INNER_CMD \" &"
  CMD="$CMD $INNER_CMD > $FN_EXEC_OUT 2>&1 & \" &"

  echo
  echo [%] $CMD
  echo

  ## 2.2 evaluate command
  if $DO_EVAL
  then
    eval $CMD
  fi
done

sleep 5



# 4. launch tensorboard daemon
for ip in ${EXEC_DAEMON_IPS[@]}
do
  ## 3.1 construct command
  INNER_CMD="nohup tensorboard --logdir ./logs --host $ip --port $TB_PORT "


  CMD="ssh $ACCOUNT@$ip  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
  CMD="$CMD cd $EXEC_DIR; "
  # CMD="$CMD $INNER_CMD \" &"
  CMD="$CMD $INNER_CMD > $FN_TB_OUT 2>&1 & \" &"

  echo
  echo [%] $CMD
  echo

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
  # python run.py --mode train --map doan --target "SA 101,SA 104" --action offset --epoch
# fi
