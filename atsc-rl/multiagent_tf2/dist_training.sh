#!/bin/bash
#
##
## script control
##
DEFAULT_OPERATION="usage"
OPERATION=${1:-$DEFAULT_OPERATION}
OPERATION=${OPERATION,,}

OP_SIMULATE="simulate" # do simulation with fixed signal to get ground zero performance
OP_TRAIN="train" # do distributed training
OP_TENSORBOARD="tensorboard" # launch tensorboard daemon
OP_MONITOR="monitor"  # check whether processes ifor distributed training are alive
OP_TERMINATE="terminate" # do terminate forcely
OP_CLEAN="clean" # do clean : remove daemon dump file (i.e.,  zz.out.*)
OP_CLEAN_ALL="clean-all" # remove files which were generated when we do training
OP_USAGE="usage" # show usage

DO_EVAL=true # whether do execution or not;  do evaluate commands if true, otherwise just dump commands

display_usage() {
  echo
  echo "[%] dist_util.py [simulate|train|tensorboard|monitor|terminate|clean]"
  echo "        simulate : do with fixed traffic signal to get ground zero performance"
  echo "        train : do distributed training"
  echo "        tensorboard : do launch tensorboard daemon"
  echo "        monitor : check whether processes for distributed training are alive "
  echo "        terminate : do terminate all processes for distributed training"
  echo "        clean : remove daemon dump log file such as zz.out.ctrl/exec/tb"
  echo "        clean-all : remove some files which were generated when we do training such as zz.out.*, logs, model, output/train, output/test, scenario history file,..."
  echo
}


if [ "$OPERATION" == "$OP_USAGE" ]; then
  display_usage
  echo "in usage"
  exit
fi


#
# 0. set parameters
#
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
  FN_PREFIX="zz.out"
  FN_POSTFIX=`date +"%F-%H-%M-%S"`

  ###--- for control daemon
  FN_CTRL_OUT="${FN_PREFIX}.ctrl.${FN_POSTFIX}"

  ###--- for execution daemon
  FN_EXEC_OUT="${FN_PREFIX}.exec.${FN_POSTFIX}"

  ###--- for tensorboard
  FN_TB_OUT="${FN_PREFIX}.tb.${FN_POSTFIX}"


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



#
# 1. do something : simulate, train, tensorboard, monitor, terminate, clean
#

#
#-- 1.1 simulate : get the performance before doing reinforcement learning as a base perdormance
#
if [ "$OPERATION" == "$OP_SIMULATE" ]
then

  ## (1) construct command
  INNER_CMD="SALT_HOME=$SALT_HOME nohup python $RL_PROG --mode simulate --scenario-file-path $RL_SCENARIO_FILE_PATH "
  INNER_CMD="$INNER_CMD --map $RL_MAP --target-TL '$RL_TARGET' --method $RL_METHOD "
  INNER_CMD="$INNER_CMD --state $RL_STATE --action $RL_ACTION --reward-func $RL_REWARD "

  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
  CMD="$CMD cd $EXEC_DIR; "
  CMD="$CMD $INNER_CMD \" "
  echo [%] $CMD

  ## (2) evaluate command
  if $DO_EVAL
  then
    eval $CMD
  fi

  #  python run.py --mode simulate --map $RL_MAP --target-TL $RL_TARGET --method $RL_METHOD --state $RL_STATE  \
  #         --action $RL_ACTION --reward-func $RL_REWARD
  echo
  echo
  echo "Simulation with fixed signal to get ground zero performance was done."
  echo "So base performance with fixed signal was gathered."


#
#-- 1.2 train : launch process for distributed training
#
elif [ "$OPERATION" == "$OP_TRAIN" ]
then
  # (1) execute controller daemon
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
  CMD="$CMD $INNER_CMD > $FN_CTRL_OUT 2>&1 & \" &"

  echo
  echo [%] $CMD
  echo

  ## (1.2) evaluate command
  if $DO_EVAL
  then
    eval $CMD
  fi

  sleep 5


  # (2) launch execution daemon
  for ip in ${EXEC_DAEMON_IPS[@]}
  do
    ## (2.1) construct command
    INNER_CMD="SALT_HOME=$SALT_HOME nohup python $EXEC_DAEMON --ip-addr $CTRL_DAEMON_IP --port $PORT "

    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
    CMD="$CMD cd $EXEC_DIR; "
    # CMD="$CMD $INNER_CMD \" &"
    CMD="$CMD $INNER_CMD > $FN_EXEC_OUT 2>&1 & \" &"
  
    echo
    echo [%] $CMD
    echo

    ## (2.2) evaluate command
    if $DO_EVAL
    then
      eval $CMD
    fi
  done

  sleep 3

#
#-- 1.3 launch tensorboard daemon
#
elif [ "$OPERATION" == "$OP_TENSORBOARD" ]
then
  for ip in ${EXEC_DAEMON_IPS[@]}
  do
    ## (1) construct command
    INNER_CMD="nohup tensorboard --logdir ./logs --host $ip --port $TB_PORT "


    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
    CMD="$CMD cd $EXEC_DIR; "
    CMD="$CMD $INNER_CMD > $FN_TB_OUT 2>&1 & \" &"

    echo
    echo [%] $CMD
    echo

    ## (2) evaluate command
    if $DO_EVAL
    then
      eval $CMD
    fi
  done


#
#-- 1.4 monitor : check whether training processes are running or not
#
elif [ "$OPERATION" == "$OP_MONITOR" ]
then
  ## (1) ctrl daemon
  echo "##"
  echo "## " ${CTRL_DAEMON_IP}
  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD ps -def | grep $CTRL_DAEMON | grep $PORT "
  echo [%] $CMD
  eval $CMD
  echo

  ## (2) exec node
  for ip in ${EXEC_DAEMON_IPS[@]}
  do
    echo "##"
    echo "## " ${ip}
    ### exec daemon
    ###--- construct command
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD ps -def | grep $EXEC_DAEMON | grep $PORT "
    echo [%] $CMD
    eval $CMD
    echo


    ### rl prog
    ### construct command
    RESULT_DIR_POSTFIX=${RL_ACTION}_${RL_REWARD}_${EXP_OPTION}
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD ps -def | grep $RL_PROG | grep $TODAY | grep $RESULT_DIR_POSTFIX "
    echo [%] $CMD
    eval $CMD
    echo


    ### tensorboard
    ###--- construct command
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD ps -def | grep tensorboard | grep $TB_PORT "
    echo [%] $CMD
    eval $CMD
    echo
  done


#-- 1.5 terminate process forcely using kill command
elif [ "$OPERATION" == "$OP_TERMINATE" ]
then
  ## (1) ctrl daemon
  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD ps -def | grep $CTRL_DAEMON | grep $PORT | awk '{print $"
  CMD="${CMD}2}'"

  pid=`eval $CMD`
  if [[ -n "$pid" ]] ; then
    CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
    CMD="$CMD kill -9 $pid"
    echo $CMD ... terminate $CTRL_DAEMON
    eval $CMD
  fi

  ## (2) exec node
  for ip in ${EXEC_DAEMON_IPS[@]}
  do
    ### exec daemon
    ###--- construct command
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD ps -def | grep $EXEC_DAEMON | grep $PORT | awk '{print $"
    CMD="${CMD}2}' "

    pid=`eval $CMD`
    if [[ -n "$pid" ]] ; then
      CMD="ssh $ACCOUNT@$ip  "
      CMD="$CMD kill -9 $pid"
      echo $CMD  ... terminate $EXEC_DAEMON
      eval $CMD
    fi


    ### rl prog
    ### construct command
    RESULT_DIR_POSTFIX=${RL_ACTION}_${RL_REWARD}_${EXP_OPTION}
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD ps -def | grep $RL_PROG | grep $TODAY | grep $RESULT_DIR_POSTFIX| awk '{print $"
    CMD="${CMD}2}' "

    pid=`eval $CMD`
    if [[ -n "$pid" ]] ; then
      CMD="ssh $ACCOUNT@$ip  "
      CMD="$CMD kill -9 $pid"
      echo $CMD ... terminate $RL_PROG
      eval $CMD
    fi


    ### tensorboard
    ###--- construct command
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD ps -def | grep tensorboard | grep $TB_PORT | awk '{print $"
    CMD="${CMD}2}' "

    pid=`eval $CMD`
    if [[ -n "$pid" ]] ; then
      CMD="ssh $ACCOUNT@$ip  "
      CMD="$CMD kill -9 $pid"
      echo $CMD ... terminate tensorboard
      eval $CMD
    fi
  done


#
#-- 1.6 clean : remove daemon dump log file such as zz.out.ctrl/exec/tb
#
elif [ "$OPERATION" == "$OP_CLEAN" ]
then
  ## (0) set to delete
  TO_DELETE=`echo ${FN_PREFIX}.*`

  ## (1) ctrl daemon
  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD 'cd $EXEC_DIR; rm -rf ${TO_DELETE} ' "
  #CMD="$CMD 'cd $EXEC_DIR; rm -rf ${FN_PREFIX}.*'"
  echo [%] $CMD
  eval $CMD

  ## (2) exec node
  for ip in ${EXEC_DAEMON_IPS[@]}
  do
    ### exec daemon
    ###--- construct command
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD 'cd $EXEC_DIR; rm -rf ${TO_DELETE} ' "
    echo [%] $CMD
    eval $CMD
  done



#
#-- 1.7 clean-all : remove some files which were generated when we do training 
#                   such as zz.out.*, zz.optimal_model_info*, logs, model,
#                            output/train, output/test, scenario history file,...
#
elif [ "$OPERATION" == "$OP_CLEAN_ALL" ]
then
  ## (0) set to delete
  TO_DELETE=`echo ${FN_PREFIX}.* zz.optimal_model_info*  ./logs  ./model `
  TO_DELETE=`echo ${TO_DELETE} ./output/train ./output/test $RL_SCENARIO_FILE_PATH/data `

  ## (1) ctrl daemon
  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD 'cd $EXEC_DIR; rm -rf $TO_DELETE'"
  echo [%] $CMD
  eval $CMD

  ## (2) exec node
  for ip in ${EXEC_DAEMON_IPS[@]}
  do
    ### exec daemon
    ###--- construct command
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD 'cd $EXEC_DIR; rm -rf $TO_DELETE'"
    echo [%] $CMD
    eval $CMD
  done

#
#-- error : entered argument is not valid
#
else
  echo $OPERATION is not valid

  display_usage
fi
