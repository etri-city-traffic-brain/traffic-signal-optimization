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
OP_TEST="test" # do test with trained model
OP_TRAINING_INFO="show-training-info"   # show training info such as target TLs
OP_USAGE="usage" # show usage

DO_EVAL=true # true # whether do execution or not;  do evaluate commands if true, otherwise just dump commands

display_usage() {
  echo
  echo "[%] dist_training.py OPERATION [START-DAY] [TEST-MODEL-NUMBER] "
  echo "        OPERATION : one of [simulate|train|tensorboard|monitor|terminate|clean|clean-all|show-result|test] "
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
  echo "            test : do test with trained model"
  echo "                      You should check START-DAY value"
  echo "            show-training-info : dump training info such as target TLs"
  echo ""
  echo "        START-DAY : start day of training; yymmdd;"
  echo "                    You should pass this value which indicates the day training was started."
  echo "                    valid if operation is one of [monitor|terminate|show-result|test] "
  echo ""
  echo "        TEST-MODEL-NUMBER : model number to do test"
  echo "                    valid if operation is one of [test] "
  echo ""
  echo "    ex., "
  echo "        [%] dist_training.sh train "
  echo "        [%] dist_training.sh terminate 220923 "
  echo "        [%] dist_training.sh test 220923 15 "
}

if [ "$OPERATION" == "$OP_USAGE" ]; then
  display_usage
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
  CTRL_DAEMON_IP="129.254.182.172" # "129.254.184.123" "129.254.182.176" "129.254.184.123"  # 101.79.1.126

  ###--- ip address of nodes to run execution daemon
  EXEC_DAEMON_IPS=(
                    "129.254.182.172"
                    "129.254.184.184"
                    "129.254.184.123"
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
  # 129.254.182.172   uniq9
  # 129.254.182.173   uniq10

  ###--- number of execution daemon
  NUM_EXEC_DAEMON=${#EXEC_DAEMON_IPS[@]}

  ###--- port to communicate btn ctrl daemon and exec daemon
  PORT=2727 #2727 3001  3101  3201  3301

  ###--- port for tensorboard
  TB_PORT=6006 #6006 7001 7101 7201 7301

  ###--- directory for traffic signal optimization(TSO) execution
  ###    You must deploy the code for execution to the same location specified below on all nodes.
  #EXEC_DIR=/home/tsoexp/z.uniq/traffic-signal-optimization/atsc-rl/multiagent_tf2
  #EXEC_DIR="/home/tsoexp/PycharmProjects/traffic-signal-optimization/atsc-rl/multiagent_tf2.0"
  EXEC_DIR=/home/tsoexp/z.uniq/2023.dev/multiagent_tf2


  ###--- conda environment for TSO
  CONDA_ENV_NAME="UniqOpt.p3.8.v2" # "UniqOpt.p3.8" "UniqOpt.p3.8.v2" "opt"
  ACTIVATE_CONDA_ENV="source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate $CONDA_ENV_NAME "

  ###-- libsalt path
  SALT_HOME=/home/tsoexp/z.uniq/2023.dev/traffic-simulator

  #######
  ## exec program
  ##
  ###--- control daemon for distributed training
  CTRL_DAEMON="DistCtrlDaemon.py"

  ###--- execution daemon for distributed training
  EXEC_DAEMON="DistExecDaemon.py"

  ###--- whether do execute LDT in ExecDaemon(in a node) parallelly or not
  ###    use this flag for comparison
  DO_PARALLEL="true"

  ###-- reinforcement learning main
  RL_PROG="run_dist_considered.py" #"run.py"

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

  ###--- for test
  FN_TEST_OUT="${FN_PREFIX}.test.${FN_POSTFIX}"


  #######
  ## Reinforcement Learning related parameters
  ##
  ###--- to access simulation scenario file(relative path)
  RL_SCENARIO_FILE_PATH="data/envs/salt"

  ###--- name of map to simulate
  RL_MAP="dj200" # one of { doan, cdd1, cdd2, cdd3, sa_1_6_17, dj_all, dj200 }


  ###-- set target to train
  if [ "$RL_MAP" == "doan" ]
  then
    ###--- target to train
    RL_TARGET="SA 101, SA 104, SA 107, SA 111" # SA 101,SA 104,SA 107,SA 111"
  elif [ "$RL_MAP" == "cdd1" ]  # 51 TS
  then
    ###--- target to train
    RL_TARGET="SA 72, SA 13"
  elif [ "$RL_MAP" == "cdd2" ]  # 59 TS
  then
    ###--- target to train
    RL_TARGET="SA 28, SA 32, SA 55, SA 56, SA 101"
  elif [ "$RL_MAP" == "cdd3" ]  # 53 TS
  then
    ###--- target to train
    RL_TARGET="SA 1, SA 6, SA 17, SA 61"
  elif [ "$RL_MAP" == "sa_1_6_17" ]
  then
    ###--- target to train
    RL_TARGET="SA 1, SA 6, SA 17"  # SA 1, SA 6, SA 17
  elif [ "$RL_MAP" == "dj200" ]
  then
    ###--- target to train
    #RL_TARGET="SA 3, SA 28, SA 101, SA 37, SA 38, SA 1, SA 102, SA 104, SA 33, SA 30"
    RL_TARGET="SA 3, SA 28, SA 101, SA 6, SA 41, SA 37, SA 38, SA 1, SA 102, SA 104, SA 33, SA 30"
    #RL_TARGET="SA 3, SA 28, SA 101, SA 6, SA 41, SA 20, SA 37, SA 38, SA 9, SA 1, SA 57, SA 102, SA 104, SA 98, SA 8, SA 33, SA 59, SA 30"
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
  RL_STATE="vd" # v, d, vd, vdd
  RL_ACTION="gt"  # offset, gr, gro, kc, gt
  RL_REWARD="wq"  # wq, cwq, pn, wt, tt

  ###--- training epoch
  RL_EPOCH=20	# 200

  ###--- interval for model saving : how open save model
  RL_MODEL_SAVE_PERIOD=1


  ###-- network-size
  NETWORK_SIZE="1024,512,512,512,512" # string of comma separated integer values are expected

  ###-- model : learning rate
  RL_MODEL_ACTOR_LR=0.0001
  RL_MODEL_CRITIC_LR=0.0001

  ###--- replay memory length
  RL_MODEL_MEM_LEN=100  # default 1000
  FORGET_RATIO=0.5 # default 0.8  .. RL_MODEL_MEM_LEN * (1-FORGET_RATIO) experiences are used to update model


  ###--- number of env when we use to train an agent; it is to increase experience
  ###     NUM_CONCURRENT_ENV environment process will be created
  NUM_CONCURRENT_ENV=5  # 10

  ##--- maximum number of simulations for learning using generated environment process;
  ##    it is to avoid memory related problem
  MAX_RUN_WITH_AN_ENV_PROCESS=50 #100

  COMP_TOTAL_ONLY=True  # compare total only when we do compare result; for fast comparison

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
#-- 1.1 simulate : get the performance before doing reinforcement learning as a base performance
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

  #  python $RL_PROG --mode simulate --map $RL_MAP --target-TL $RL_TARGET --method $RL_METHOD --state $RL_STATE  \
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
  #    -u : forcely flush print  ... ref.https://www.delftstack.com/ko/howto/python/python-print-flush/
  #INNER_CMD="SALT_HOME=$SALT_HOME nohup python $CTRL_DAEMON --port $PORT --num-of-learning-daemon $NUM_EXEC_DAEMON "
  INNER_CMD="SALT_HOME=$SALT_HOME nohup python -u $CTRL_DAEMON --port $PORT --num-of-learning-daemon $NUM_EXEC_DAEMON "
  INNER_CMD="$INNER_CMD --validation-criteria $IMPROVEMENT_GOAL "
  INNER_CMD="$INNER_CMD --num-of-optimal-model-candidate $NUM_OF_OPTIMAL_MODEL_CANDIDATE "
  INNER_CMD="$INNER_CMD --cumulative-training $CUMULATIVE_TRAINING "
  INNER_CMD="$INNER_CMD --model-store-root-path $MODEL_STORE_ROOT_PATH/$RESULT_DIR "
  INNER_CMD="$INNER_CMD --copy-simulation-output $COPY_SIMULATION_OUTPUT "

  INNER_CMD="$INNER_CMD --scenario-file-path $RL_SCENARIO_FILE_PATH "
  INNER_CMD="$INNER_CMD --map $RL_MAP --target-TL '$RL_TARGET' --method $RL_METHOD "
  INNER_CMD="$INNER_CMD --state $RL_STATE --action $RL_ACTION --reward-func $RL_REWARD "
  INNER_CMD="$INNER_CMD --model-save-period $RL_MODEL_SAVE_PERIOD --epoch $RL_EPOCH "
  INNER_CMD="$INNER_CMD --network-size $NETWORK_SIZE "
  INNER_CMD="$INNER_CMD --a-lr $RL_MODEL_ACTOR_LR --c-lr $RL_MODEL_CRITIC_LR "
  INNER_CMD="$INNER_CMD --mem-len $RL_MODEL_MEM_LEN --mem-fr $FORGET_RATIO "
  INNER_CMD="$INNER_CMD --num-concurrent-env $NUM_CONCURRENT_ENV "
  INNER_CMD="$INNER_CMD --max-run-with-an-env-process $MAX_RUN_WITH_AN_ENV_PROCESS "
  INNER_CMD="$INNER_CMD --comp-total-only $COMP_TOTAL_ONLY "



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
    # INNER_CMD="SALT_HOME=$SALT_HOME nohup python $EXEC_DAEMON --ip-addr $CTRL_DAEMON_IP --port $PORT --do-parallel $DO_PARALLEL"
    INNER_CMD="SALT_HOME=$SALT_HOME nohup python -u $EXEC_DAEMON --ip-addr $CTRL_DAEMON_IP --port $PORT --do-parallel $DO_PARALLEL"

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


  # (3) copy this script
  CMD="cp $0 $MODEL_STORE_ROOT_PATH/$RESULT_DIR"

  ## (3.1) evaluate command
  echo
  echo [%] $CMD
  echo

  ## (3.2) evaluate command
  if $DO_EVAL
  then
    eval $CMD
  fi

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
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD ps -def | grep $RL_PROG | grep $START_DAY | grep $RESULT_DIR_LEAF "
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

  echo ""
  echo ""
  echo "You can not find $RL_PROG process with this script when we do first round because infer-mode-path is not set. "

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



  ## (1) exec node
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
    CMD="ssh $ACCOUNT@$ip  "
    CMD="$CMD ps -def | grep $RL_PROG | grep $START_DAY | grep $RESULT_DIR_LEAF | awk '{print $"
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


  ## (2) ctrl daemon
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

  echo "You can not find $RL_PROG process with this script when we do first round beacuse infer-mode-path is not set. "


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
#-- 1.9 test : test with trained model
#
elif [ "$OPERATION" == "$OP_TEST" ]
then
  START_DAY=$2

  DEFAULT_TEST_MODEL_NUM="0"
  TEST_MODEL_NUM=${3:-$DEFAULT_TEST_MODEL_NUM}

  RESULT_DIR_LEAF=${RL_MAP}_${RL_STATE}_${RL_ACTION}_${RL_REWARD}_${EXP_OPTION} # ex., doan_vdd_gr_wq_all
  RESULT_DIR=${START_DAY}/${RESULT_DIR_LEAF} # ex., 220713/doan_gr_wq_all

  INNER_CMD="SALT_HOME=$SALT_HOME nohup python $RL_PROG "
  INNER_CMD="$INNER_CMD  --mode test "
  INNER_CMD="$INNER_CMD --scenario-file-path $RL_SCENARIO_FILE_PATH "
  INNER_CMD="$INNER_CMD --map $RL_MAP --target-TL '$RL_TARGET' "
  INNER_CMD="$INNER_CMD --method $RL_METHOD "
  INNER_CMD="$INNER_CMD --state $RL_STATE --action $RL_ACTION --reward-func $RL_REWARD "
  INNER_CMD="$INNER_CMD --epoch 1 "
  INNER_CMD="$INNER_CMD --network-size $NETWORK_SIZE "
  INNER_CMD="$INNER_CMD --a-lr $RL_MODEL_ACTOR_LR --c-lr $RL_MODEL_CRITIC_LR "
  INNER_CMD="$INNER_CMD --mem-len $RL_MODEL_MEM_LEN --mem-fr $FORGET_RATIO "

  INNER_CMD="$INNER_CMD --model-num  $TEST_MODEL_NUM "
  INNER_CMD="$INNER_CMD --infer-model-path  $MODEL_STORE_ROOT_PATH/$RESULT_DIR "
  INNER_CMD="$INNER_CMD --result-comp True "



  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
  #CMD="$CMD cd $CTRL_DIR; "
  CMD="$CMD cd $EXEC_DIR; "
  CMD="$CMD $INNER_CMD > $FN_TEST_OUT 2>&1 & \" &"


  echo ""
  echo [%] $CMD
  echo [%] $CMD > $FN_TEST_OUT
  echo

  ## (1.2) evaluate command
  if $DO_EVAL
  then
    eval $CMD
  fi

  echo "The trained model whose model number is $TEST_MODEL_NUM is now being evaluated"
  echo "    Stored location of the trained model : $MODEL_STORE_ROOT_PATH/$RESULT_DIR"
  echo ""
  echo "You should make sure that performance of ground zero was collected for comparison with the performance of a trained model."
  echo "Visit following directory to see Test Results"
  #echo "    $CTRL_DIR/output/test  "
  echo "    $EXEC_DIR/output/test  "


#
#-- 1.10 show-training-info : show training info such as target TLs
#
elif [ "$OPERATION" == "$OP_TRAINING_INFO" ]
then
  ####
  #### show training target
  echo ""

  SHOW_PROG="./tools/ShowTargetTL.py"
  INNER_CMD="SALT_HOME=$SALT_HOME nohup python $SHOW_PROG "
  INNER_CMD="$INNER_CMD --scenario-file-path $RL_SCENARIO_FILE_PATH "
  INNER_CMD="$INNER_CMD --map $RL_MAP --target-TL '$RL_TARGET' "

  CMD="ssh $ACCOUNT@$CTRL_DAEMON_IP  "
  CMD="$CMD \" $ACTIVATE_CONDA_ENV; "
  CMD="$CMD cd $EXEC_DIR; "
  CMD="$CMD $INNER_CMD \""

  #echo [%] $CMD
  eval $CMD
  echo


  ####
  #### show path which stores trained models
  echo ""
  START_DAY=$2

  RESULT_DIR=${START_DAY}/${RESULT_DIR_LEAF} # ex., 220713/doan_gr_wq_all

  echo "You can see trained model by visiting following directory"
  echo "     $MODEL_STORE_ROOT_PATH/$RESULT_DIR at $ACCOUNT@$CTRL_DAEMON_IP "

#
#-- error : entered argument is not valid
#
else
  echo $OPERATION is not valid

  display_usage
fi
