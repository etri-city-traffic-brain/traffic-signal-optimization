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
OP_SHOW_RESULT="show-result" # dump training result by showing the calculated improvement rate of each round
OP_USAGE="usage" # show usage

DO_EVAL=true # whether do execution or not;  do evaluate commands if true, otherwise just dump commands

display_usage() {
  echo
  echo "[%] dist_util.py OPERATION [START-DAY] "
  echo "        OPERATION : one of [simulate|train|tensorboard|monitor|terminate|clean] "
  echo "            simulate : do with fixed traffic signal to get ground zero performance"
  echo "            train : do distributed training"
  echo "            tensorboard : do launch tensorboard daemon"
  echo "            monitor : check whether processes for distributed training are alive"
  echo "                      You should check START-DAY value"
  echo "            terminate : do terminate all processes for distributed training"
  echo "                      You should check START-DAY value"
  echo "            clean : remove daemon dump log file such as zz.out.ctrl/exec/tb"
  echo "            clean-all : remove some files which were generated when we do training "
  echo "                        such as zz.out.*, logs, model, output/train, output/test, scenario history file,..."
  echo "            show-result : dump training result by showing the calculated improvement rate of each round"
  echo "                      You should check START-DAY value"
  echo "        START-DAY : start day of training; yymmdd;"
  echo "                    You should pass this value which indicates the day training was started."
  echo "                    valid if operation is one of [monitor|terminate|show-result] "
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
  CTRL_DAEMON_IP="129.254.182.176"  # 101.79.1.126

  ###--- directory for traffic signal optimization(TSO) execution : Controller
  #CTRL_DIR="/home/tsoexp/z.uniq/traffic-signal-optimization/atsc-rl/multiagent_tf2.0"
  CTRL_DIR="/home/tsoexp/PycharmProjects/traffic-signal-optimization/atsc-rl/multiagent_tf2.0"

  ###--- ip address of nodes to run execution daemon
  ###    EXEC_DAEMON_IPS should pair with EXEC_DIRS
  EXEC_DAEMON_IPS=(
                    "129.254.182.176"
                    "129.254.182.176"
                    "129.254.182.176"
                  )
  #
  # 129.254.184.239		uniq1
  # 129.254.184.241		uniq2
  # 129.254.184.123		uniq3
  # 129.254.184.184		uniq4
  # 129.254.184.238		uniq5
  # 129.254.184.248		uniq6
  # 129.254.184.53		uniq7
  # 129.254.184.54		uniq8
  # 129.254.182.176		6th

  ###--- directories for traffic signal optimization(TSO) execution : Executor
  ###    EXEC_DIRS should pair with EXEC_DAEMON_IPS
  #EXEC_DIR=/home/tsoexp/z.uniq/traffic-signal-optimization/atsc-rl/multiagent_tf2
  EXEC_DIRS=(
	     "/home/tsoexp/PycharmProjects/traffic-signal-optimization/atsc-rl/multiagent_tf2.0"
	     "/home/tsoexp/PycharmProjects/traffic-signal-optimization/atsc-rl/multiagent_tf2.1"
	     "/home/tsoexp/PycharmProjects/traffic-signal-optimization/atsc-rl/multiagent_tf2.2"
           )

  ###--- number of execution daemon
  NUM_EXEC_DAEMON=${#EXEC_DAEMON_IPS[@]}

  ###-- make uniq ip address of nodes to run exec daemon
  ###   this will be used to execute some operations such as terminate/clean/clean-all/monitor
  TMP_FILE="/tmp/__IPS"
  rm -rf $TMP_FILE
  touch $TMP_FILE
  echo ${CTRL_DAEMON_IP} >> $TMP_FILE
  for ip in ${EXEC_DAEMON_IPS[@]}
  do
    echo ${ip} >> $TMP_FILE
  done
  UNIQ_EXP_IPS=`cat $TMP_FILE | sort | uniq `
  #echo UNIQ_EXP_IPS $UNIQ_EXP_IPS
  rm -rf $TMP_FILE

  ###--- port to communicate btn ctrl daemon and exec daemon
  PORT=2727 #2727 3001  3101  3201  3301

  ###--- port for tensorboard 
  ###--- TB_PORTS should pair with EXEC_DAEMON_IPS
  #TB_PORT=6006 #6006 7001 7101 7201 7301
  TB_PORTS=(
	    6006
	    6016
	    6026
	  )

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
  #RL_MAP="doan" # one of { doan, sa_1_6_17, dj_all }
  RL_MAP="sa_1_6_17" # one of { doan, sa_1_6_17, dj_all }
  #RL_MAP="dj_all" # one of { doan, sa_1_6_17, dj_all }

  ###-- set target to train
  if [ "$RL_MAP" == "doan" ]
  then
    ###--- target to train
    RL_TARGET="SA 101, SA 104, SA 107, SA 111" # SA 101,SA 104,SA 107,SA 111"
  elif [ "$RL_MAP" == "sa_1_6_17" ]
  then
    ###--- target to train
    RL_TARGET="SA 1, SA 6, SA 17"  # SA 1, SA 6, SA 17
  elif [ "$RL_MAP" == "dj_all" ]
  then
    ###--- target to train
    ####-- candidate1
    #RL_TARGET="SA 13, SA 72"
       # 51 TLs = SA 13(29 TLs) + SA 72(23 TLs)

    ####-- candidate2
    RL_TARGET="SA 56, SA 101, SA 28, SA 55, SA 32" 
       # 59 TLs = SA 56(15 TLs) + SA 101(10 TLs) + SA 28(12 TLs) + SA 55(10 TLs) + SA 32(12 TLs)" 

    ####-- candidate3
    #RL_TARGET="SA 1, SA 6, SA 17, SA 61"
       # 53 TLs = SA 1(13 TLs) + SA 6(11 TLs) + SA 17(19 TLs) + SA 61(10 TLs)
  fi

  ###--- RL method
  RL_METHOD="sappo"

  ###--- state, action, reward for RL
  RL_STATE="vdd" # v, d, vd, vdd
  RL_ACTION="gro"  # offset, gr, gro, kc
  RL_REWARD="cwq"  # wq, cwq, pn, wt, tt

  ###--- training epoch
  RL_EPOCH=2	# 200

  ###--- interval for model saving : how open save model
  RL_MODEL_SAVE_PERIOD=1

  ###--- replay memory length
  RL_MODEL_MEM_LEN=500  # default 1000
  FORGET_RATIO=0.5 # default 0.8  .. RL_MODEL_MEM_LEN * (1-FORGET_RATIO) experiences are used to update model

  #######
  ## distributed Reinforcement Learning related parameters
  ##
  ###--- training improvement goal
  IMPROVEMENT_GOAL=20.0

  ###-- shared directory;
  ###-- should be accessed by all ctrl/exec daemon
  MODEL_STORE_ROOT_PATH="/home/tsoexp/share/dist_training"

  ###--- directory to save training result
  START_DAY=`date +"%g%m%d"`  # 220701

  EXP_OPTION="all" # all , sa101, sa6, rm
  #
  # SA 101, SA 104 ==> SA101,SA104 ==>SA101_SA104
  #EXP_OPTION="${RL_TARGET// /}" # remove blank
  #EXP_OPTION="${EXP_OPTION//,/_}" # replace comma(,) to underscore(_)

  RESULT_DIR_LEAF=${RL_MAP}_${RL_STATE}_${RL_ACTION}_${RL_REWARD}_${EXP_OPTION} # ex., doan_vdd_gr_wq_all
  RESULT_DIR=${START_DAY}/${RESULT_DIR_LEAF} # ex., 220713/doan_gr_wq_all


  ###--- number of optimal model candidate
  NUM_OF_OPTIMAL_MODEL_CANDIDATE=10

  ###--- whether do copy simulation output file or not : PeriodicOutput, rl_phase_reward_output,
  COPY_SIMULATION_OUTPUT="yes" # yes, true, t, TRUE, ... no, False, f

  ###-- whether do cumulative training or not : model, replay memory
  CUMULATIVE_TRAINING="True"

  DIST_RESULT_FILE="zz.dist_learning_history.csv" # contains improved performance rate of each round ; distributed training history
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
  CMD="$CMD cd $CTRL_DIR; "
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
  INNER_CMD="$INNER_CMD --mem-len $RL_MODEL_MEM_LEN --mem-fr $FORGET_RATIO "

  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
  CMD="$CMD cd $CTRL_DIR; "
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
  for i in $(seq 0 `expr $NUM_EXEC_DAEMON - 1 ` )
  do
    # echo "#### i=$i  num_exec_daemon=$NUM_EXEC_DAEMON "

    ## (2.1) construct command
    INNER_CMD="SALT_HOME=$SALT_HOME nohup python $EXEC_DAEMON --ip-addr $CTRL_DAEMON_IP --port $PORT "

    CMD="ssh $ACCOUNT@${EXEC_DAEMON_IPS[$i]}  "
    CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
    CMD="$CMD cd ${EXEC_DIRS[$i]}; "
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
  for i in $(seq 0 `expr $NUM_EXEC_DAEMON - 1 ` )
  do
    ## (1) construct command
    INNER_CMD="nohup tensorboard --logdir ./logs --host ${EXEC_DAEMON_IPS[$i]} --port ${TB_PORTS[$i]} "


    CMD="ssh $ACCOUNT@${EXEC_DAEMON_IPS[$i]}  "
    CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
    CMD="$CMD cd ${EXEC_DIRS[$i]}; "
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
  ## (0) check command is valid and set START_DAY value
  if [ $# -ne 2 ]
  then
    echo "invalid argument...The number of argument should be 2. "
    display_usage
    exit
  fi

  START_DAY=$2

  ## (1) ctrl daemon
  echo "##"
  echo "## " ${CTRL_DAEMON_IP}
  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD ps -def | grep $CTRL_DAEMON | grep $PORT "
  echo [%] $CMD
  if $DO_EVAL
  then
    eval $CMD
  fi
  echo


  ## (2) exec daemon & rl prog
  for ip in ${UNIQ_EXP_IPS[@]}
  do
      echo "##"
      echo "## " $ip
      ### for exec daemon
      ###--- construct command
      CMD="ssh $ACCOUNT@$ip  "
      CMD="$CMD ps -def | grep $EXEC_DAEMON | grep $PORT "

      echo [%] $CMD
      if $DO_EVAL
      then
        eval $CMD
      fi
      echo


      ### for rl prog
      ### construct command
      CMD="ssh $ACCOUNT@$ip   "
      CMD="$CMD ps -def | grep $RL_PROG | grep $START_DAY | grep $RESULT_DIR_LEAF "
      echo [%] $CMD
      if $DO_EVAL
      then
        eval $CMD
      fi
      echo
  done

  ## (3) tensorboard
  for i in $(seq 0 `expr $NUM_EXEC_DAEMON - 1 ` )
  do
    ###--- construct command
    CMD="ssh $ACCOUNT@${EXEC_DAEMON_IPS[$i]}  "
    CMD="$CMD ps -def | grep tensorboard | grep ${TB_PORTS[$i]} "
    echo [%] $CMD
    if $DO_EVAL
    then
      eval $CMD
    fi
    echo
  done

  echo "You can not find run.py process with this script when we do first round beacuse infer-mode-path is not set. "

#-- 1.5 terminate process forcely using kill command
elif [ "$OPERATION" == "$OP_TERMINATE" ]
then
  ## (0) check command is valid and set START_DAY value
  if [ $# -ne 2 ]
  then
    echo "invalid argument...The number of argument should be 2. "
    display_usage
  fi

  START_DAY=$2



  ## (1) exec daemon & rl prog
  for ip in ${UNIQ_EXP_IPS[@]}
  do
    ### exec daemon
    ###--- construct command
    CMD="ssh $ACCOUNT@${ip}  "
    CMD="$CMD ps -def | grep $EXEC_DAEMON | grep $PORT | awk '{print $"
    CMD="${CMD}2}' "


    if $DO_EVAL
    then
      pid=`eval $CMD`
    else
      echo $CMD
    fi

    if [[ -n "$pid" ]] ; then
      CMD="ssh $ACCOUNT@${ip}  "
      CMD="$CMD kill -9 $pid"
      echo $CMD  ... terminate $EXEC_DAEMON
      eval $CMD
    fi


    ### rl prog
    ### construct command
    CMD="ssh $ACCOUNT@${ip}  "
    CMD="$CMD ps -def | grep $RL_PROG | grep $START_DAY | grep $RESULT_DIR_LEAF | awk '{print $"
    CMD="${CMD}2}' "

    if $DO_EVAL
    then
      pid=`eval $CMD`
    else
      echo $CMD
    fi

    if [[ -n "$pid" ]] ; then
      CMD="ssh $ACCOUNT@${ip}  "
      CMD="$CMD kill -9 $pid"
      echo $CMD ... terminate $RL_PROG
      eval $CMD
    fi
  done

  ## (2) tensorboard
  for i in $(seq 0 `expr $NUM_EXEC_DAEMON - 1 ` )
  do
    ###--- construct command
    CMD="ssh $ACCOUNT@${EXEC_DAEMON_IPS[$i]}  "
    CMD="$CMD ps -def | grep tensorboard | grep ${TB_PORTS[$i]} | awk '{print $"
    CMD="${CMD}2}' "

    if $DO_EVAL
    then
      pid=`eval $CMD`
    else
      echo $CMD
    fi

    if [[ -n "$pid" ]] ; then
      CMD="ssh $ACCOUNT@${EXEC_DAEMON_IPS[$i]}  "
      CMD="$CMD kill -9 $pid"
      echo $CMD ... terminate tensorboard
      eval $CMD
    fi
  done


  ## (3) ctrl daemon
  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD ps -def | grep $CTRL_DAEMON | grep $PORT | awk '{print $"
  CMD="${CMD}2}'"

  if $DO_EVAL
  then
    pid=`eval $CMD`
  else
    echo $CMD
  fi

  if [[ -n "$pid" ]] ; then
    CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
    CMD="$CMD kill -9 $pid"
    echo $CMD ... terminate $CTRL_DAEMON
    eval $CMD
  fi

  echo "You can not find run.py process with this script when we do first round beacuse infer-mode-path is not set. "


#
#-- 1.6 clean : remove daemon dump log file such as zz.out.ctrl/exec/tb
#
elif [ "$OPERATION" == "$OP_CLEAN" ]
then
  ## (0) set to delete
  TO_DELETE=`echo ${FN_PREFIX}.*`

  ## (1) ctrl daemon
  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD 'cd $CTRL_DIR; rm -rf ${TO_DELETE} ' "
  #CMD="$CMD 'cd $CTRL_DIR; rm -rf ${FN_PREFIX}.*'"
  echo [%] $CMD
  eval $CMD

  ## (2) exec node
  for i in $(seq 0 `expr $NUM_EXEC_DAEMON - 1 ` )
  do
    ### exec daemon
    ###--- construct command
    CMD="ssh $ACCOUNT@${EXEC_DAEMON_IPS[$i]}  "
    CMD="$CMD 'cd ${EXEC_DIRS[$i]}; rm -rf ${TO_DELETE} ' "
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
  CMD="$CMD 'cd $CTRL_DIR; rm -rf $TO_DELETE'"
  echo [%] $CMD
  eval $CMD

  ## (2) exec node
  for i in $(seq 0 `expr $NUM_EXEC_DAEMON - 1 ` )
  do
    ### exec daemon
    ###--- construct command
    CMD="ssh $ACCOUNT@${EXEC_DAEMON_IPS[$i]}  "
    CMD="$CMD 'cd ${EXEC_DIRS[$i]}; rm -rf $TO_DELETE'"
    echo [%] $CMD
    eval $CMD
  done




#
#-- 1.8 show result : dump training result by showing the calculated improvement rate of each round
#
elif [ "$OPERATION" == "$OP_SHOW_RESULT" ]
then
  ## (0) check command is valid and set START_DAY value
  if [ $# -ne 2 ]
  then
    echo "invalid argument...The number of argument should be 2. "
    display_usage
    exit
  fi

  START_DAY=$2

  RESULT_DIR=${START_DAY}/${RESULT_DIR_LEAF} # ex., 220713/doan_gr_wq_all

  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" cat $MODEL_STORE_ROOT_PATH/$RESULT_DIR/$DIST_RESULT_FILE \" "
  echo [%] $CMD
  eval $CMD


#
#-- error : entered argument is not valid
#
else
  echo $OPERATION is not valid

  display_usage
fi
