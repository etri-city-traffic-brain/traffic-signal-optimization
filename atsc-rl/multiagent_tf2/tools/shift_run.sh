#!/bin/bash
#

ACTIVATE_CONDA_ENV_UNIQ_OPT="source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate UniqOpt.p3.8 "
eval $ACTIVATE_CONDA_ENV_UNIQ_OPT


#
# 0. set parameters
#
if [ 1 ]; then
  ###--- directory for traffic signal optimization(TSO) execution
  #EXEC_DIR=/home/tsoexp/z.uniq/traffic-signal-optimization/atsc-rl/multiagent_tf2
  #EXEC_DIR=/home/tsoexp/z.uniq/0704/multiagent_tf2.sa101.cwq
  EXEC_DIR=/home/tsoexp/z.uniq/0730.multiagent_tf2
  EXEC_DIR=/home/tsoexp/PycharmProjects/traffic-signal-optimization/atsc-rl/multiagent_tf2
  SCENARIO_PATH="data/envs/salt"

  #######
  ## env related parameters
  ##

  ###--- name of map to simulate
  RL_MAP="doan" # one of { doan, sa_1_6_17, dj_all }
  #RL_MAP="sa_1_6_17" # one of { doan, sa_1_6_17, dj_all }
  #RL_MAP="dj_all" # one of { doan, sa_1_6_17, dj_all }

  ###-- set target to train
  if [ "$RL_MAP" == "doan" ]
  then
    ###--- target to train
    RL_TARGET="SA 101" # SA 101,SA 104,SA 107,SA 111"
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
  RL_ACTION="gr"  # offset, gr, gro, kc
  RL_REWARD="cwq"  # wq, cwq, pn, wt, tt

  MODEL_NUM=180

  ###--- statistics
  ###
  ### 6 if avg(average speed), 7 avg(average travel time), 8 sum_passed_num, 9 sum_travel_time
  AVG_SPEED=6
  AVG_TRAVEL_TIME=7
  SUM_PASSED_NUM=8
  SUM_TRAVEL_TIME=9

  COMP_FIELD_1=$SUM_PASSED_NUM
  COMP_FIELD_2=$SUM_TRAVEL_TIME

  ###
  ### file name to store result of comparison
  FN_SHIFT_COMP_RESULT="zz.shift_comp_result.csv"
  FN_COMP_RESULT="zz.result_comp_s600.${MODEL_NUM}.csv"

  ###
  ### case of comparisopm : shift
  COMP_CASES=(
                    "0"    "5"    "10"   "15"   "20"   "25"   "30"   "35"    "40"   "45"
                    "50"   "55"   "60"   "70"   "80"   "90"   "100"  "150"   "300"  "600"
             )
  SHIFT_OP="inc" # inc or dec
fi


EXEC_CFG="--map ${RL_MAP} --target-TL \"$RL_TARGET\"  --state $RL_STATE --action $RL_ACTION --reward-func $RL_REWARD "
FIXED_EXEC_CMD="python run.py $EXEC_CFG --mode simulate"
RL_EXEC_CMD="python run.py $EXEC_CFG --mode test --model-num $MODEL_NUM"

cd $EXEC_DIR

echo "#fixed_time(sum_sum_passed_num;sum_sum_travel_time;PN/TT)" > $FN_SHIFT_COMP_RESULT
echo "#rl_control(sum_sum_passed_num;sum_sum_travel_time;PN/TT)" >> $FN_SHIFT_COMP_RESULT
echo "shift,fixed_time,rl_control,improvement_rate" >> $FN_SHIFT_COMP_RESULT

for trial in ${COMP_CASES[@]}
do
    echo "## shift_${trial}"
    ## 0. prepare scenario
    echo rm $EXEC_DIR/$SCENARIO_PATH/doan
    rm $EXEC_DIR/$SCENARIO_PATH/doan
    echo ln -s $EXEC_DIR/$SCENARIO_PATH/doan.${SHIFT_OP}.${trial} $EXEC_DIR/$SCENARIO_PATH/doan
    ln -s $EXEC_DIR/$SCENARIO_PATH/doan.${SHIFT_OP}.${trial} $EXEC_DIR/$SCENARIO_PATH/doan

    ## 1. get fixed-time-controlledd performance
    eval $FIXED_EXEC_CMD
    cp output/simulate/ft_phase_reward_output.txt  output/simulate/ft_phase_reward_output.txt.${SHIFT_OP}${trial}
    cp output/simulate/_PeriodicOutput.csv  output/simulate/_PeriodicOutput.${trial}.ft.csv

    ## 2. get RL-controlled performance
    eval $RL_EXEC_CMD
    cp output/test/rl_phase_reward_output.txt  output/test/rl_phase_reward_output.txt.${SHIFT_OP}${trial}
    cp output/test/_PeriodicOutput.csv  output/test/_PeriodicOutput.${trial}.rl.csv

    ## 3. analize output file
    ###-- fixed time
    #ANAL_FIXED_CMD="cat output/simulate/ft_phase_reward_output.txt.shift${trial} | awk -F , ' {sumA += $"
    #ANAL_FIXED_CMD="${ANAL_FIXED_CMD}${COMP_FIELD}; cnt += 1} END { print \"fixed time control : sum=\" sumA \",  cnt=\" cnt \", avg= \" sumA/cnt }'"
    #eval $ANAL_FIXED_CMD
    ANAL_FIXED_CMD="cat output/simulate/ft_phase_reward_output.txt.${SHIFT_OP}${trial} | awk -F , ' {sumA += $"
    ANAL_FIXED_CMD="${ANAL_FIXED_CMD}${COMP_FIELD_1}; sumB += $"
    #ANAL_FIXED_CMD="${ANAL_FIXED_CMD}${COMP_FIELD_2} } END { print sumB/sumA }'"
    ANAL_FIXED_CMD="${ANAL_FIXED_CMD}${COMP_FIELD_2} } END { print sumA \"   \" sumB \"   \" sumB/sumA }'"
    echo $ANAL_FIXED_CMD

    ANAL_FT=`eval $ANAL_FIXED_CMD`


    ###-- RL
    #ANAL_RL_CMD="cat output/test/rl_phase_reward_output.txt.shift${trial} | awk -F , ' {sumA += $"
    #ANAL_RL_CMD="${ANAL_RL_CMD}${COMP_FIELD}; cnt += 1} END { print \"RL-agent   control : sum=\" sumA \",  cnt=\" cnt \", avg= \" sumA/cnt }'"
    #eval $ANAL_RL_CMD
    ANAL_RL_CMD="cat output/test/rl_phase_reward_output.txt.${SHIFT_OP}${trial} | awk -F , ' {sumA += $"
    ANAL_RL_CMD="${ANAL_RL_CMD}${COMP_FIELD_1}; sumB += $"
    ANAL_RL_CMD="${ANAL_RL_CMD}${COMP_FIELD_2} } END { print sumA \"   \" sumB \"   \" sumB/sumA }'"
    ANAL_RL=`eval $ANAL_RL_CMD`
   
    ## 4. save it
    ###-- improvement rate
    IMP=$( echo "$ANAL_FT; $ANAL_RL" | awk '{printf "%f", ($3 - $6) / $3 }' )

    echo $trial,$ANAL_FT,$ANAL_RL,$IMP >> $FN_SHIFT_COMP_RESULT
    
    ###-- copy result file generated by RL test prog
    #echo cp $FN_COMP_RESULT ${FN_COMP_RESULT}_${trial}
    #cp $FN_COMP_RESULT ${FN_COMP_RESULT}_${trial}
    #FN_COMP_RESULT="zz.result_comp_s600.${MODEL_NUM}.csv"

    FN_COMP_RESULT_SAVE=$( echo ${FN_COMP_RESULT}.${trial} | awk -F . '{print $1 "." $2 "." $3 "_" $5 "." $4 }' )
    #echo cp $FN_COMP_RESULT $FN_COMP_RESULT_SAVE
    cp $FN_COMP_RESULT $FN_COMP_RESULT_SAVE
    echo $FN_COMP_RESULT_SAVE
done
