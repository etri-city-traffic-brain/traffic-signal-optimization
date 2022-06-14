#!/bin/bash
#conda activate p3.8
#export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
#

#TEST_DIR=/home/tsoexp/z.test/traffic-signal-optimization-for-dist/atsc-rl/multiagent_tf2
# TEST_DIR=/home/tsoexp/z.uniq/z.multiagent_tf2.copy.ing
TEST_DIR=/home/tsoexp/PycharmProjects/traffic-signal-optimization-for-dist/atsc-rl/multiagent_tf2

# for distributed
IMP_GOAL=20.0
IP_ADDR="129.254.182.176" #"101.79.1.112"
PORT=3001 # 1737  # 2727
TB_PORT=7001 # 6163 #6006
NUM_OPT_MODEL_CANDIDATE=10
#MODEL_STORE_PATH=/home/tsoexp/share/results
#MODEL_STORE_PATH=/home/tsoexp/z.local.share.results
MODEL_STORE_PATH=/home/tsoexp/share/results
COPY_SIMULATION_OUTPUT="yes"

RESULT_DIR="0611"

# for single
NUM_DAEMON=1
MAP="doan" # "sa_1_6_17"
TARGET="SA 101, SA 104, SA 107, SA 111" # "SA 1, SA 6, SA 17"
STATE="vdd"
ACTION="gro"
REWARD_FUNC="cwq"
EPOCH=1
MODEL_SAVE_PERIOD=5
EXP_OPTION="copy"


cd $TEST_DIR

echo "nohup python DistCtrlDaemon.py --port $PORT --map $MAP --target-TL "$TARGET" --num-of-learning-daemon $NUM_DAEMON --state $STATE --action $ACTION  --reward-func $REWARD_FUNC --validation-criteria $IMP_GOAL --epoch $EPOCH --model-save-period $MODEL_SAVE_PERIOD --cumulative-training True --num-of-optimal-model-candidate $NUM_OPT_MODEL_CANDIDATE --model-store-root-path $MODEL_STORE_PATH/$RESULT_DIR/${ACTION}_${REWARD_FUNC}_${EXP_OPTION} --copy-simulation-output $COPY_SIMULATION_OUTPUT > out.ctrl 2>&1 &"

nohup python DistCtrlDaemon.py --port $PORT --map $MAP --target-TL "$TARGET" \
       	--num-of-learning-daemon $NUM_DAEMON \
       	--state $STATE --action $ACTION  --reward-func $REWARD_FUNC \
	--validation-criteria $IMP_GOAL \
	--epoch $EPOCH --model-save-period $MODEL_SAVE_PERIOD \
       	--cumulative-training True \
       	--num-of-optimal-model-candidate $NUM_OPT_MODEL_CANDIDATE \
	--model-store-root-path $MODEL_STORE_PATH/$RESULT_DIR/${ACTION}_${REWARD_FUNC}_${EXP_OPTION} \
        --copy-simulation-output $COPY_SIMULATION_OUTPUT \
	> out.ctrl 2>&1 &
sleep 3

echo "nohup python DistExecDaemon.py --ip-addr $IP_ADDR --port $PORT > out.exec 2>&1 &"
nohup python DistExecDaemon.py --ip-addr $IP_ADDR --port $PORT > out.exec 2>&1 &
sleep 3

echo "nohup tensorboard --logdir ./logs --host $IP_ADDR --port $TB_PORT > out.tb 2>&1 &"
nohup tensorboard --logdir ./logs --host $IP_ADDR --port $TB_PORT > out.tb 2>&1 &

#sleep 3

ps -def | grep python | grep tsoexp | grep $PORT
