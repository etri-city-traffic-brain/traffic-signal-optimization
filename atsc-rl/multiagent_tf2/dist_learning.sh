#!/bin/bash
#
# [%] dist_learning.sh
CTRL_DAEMON_IP="129.254.182.176"
PORT=2727
ACCOUNT="tsoexp"
EXE_DAEMON_IPS=(
  "129.254.182.176"
  "129.254.184.53"
  "129.254.182.176"
)
#    "129.254.184.53"

VALIDATION_CRITERIA=6.0

NUM_EXE_DAEMON=${#EXE_DAEMON_IPS[@]}

echo python ServerDaemon.py --port $PORT --num_of_learning_daemon $NUM_EXE_DAEMON --validation_criteria $VALIDATION_CRITERIA
python ServerDaemon.py --port $PORT --num_of_learning_daemon $NUM_EXE_DAEMON --validation_criteria $VALIDATION_CRITERIA &

sleep 5

ACTIVATE_CONDA_ENV_P3_8="source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate p3.8"

FN=foo
EXEC_DIR=/tmp/CSD

for ip in ${EXE_DAEMON_IPS[@]}
do
  echo "#################" $ip

  #-- following is work well
  # echo ssh $ACCOUNT@$ip  "source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate p3.8; cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT"
  # ssh $ACCOUNT@$ip  "source /home/tsoexp/miniforge3/etc/profile.d/conda.sh; conda activate p3.8; cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" &
  ##
  echo ssh $ACCOUNT@$ip  "$ACTIVATE_CONDA_ENV_P3_8; cd $EXEC_DIR; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" &
  ssh $ACCOUNT@$ip  "$ACTIVATE_CONDA_ENV_P3_8; cd $EXEC_DIR; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" &

  #-- following is nor work well
  # echo  ssh $ACCOUNT@$ip "cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT"
  # ssh $ACCOUNT@$ip "cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" &   # error
  ##
  # echo ssh $ACCOUNT@$ip ". ~/.bashrc; cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" & # error
  # ssh $ACCOUNT@$ip ". ~/.bashrc; cd /tmp/CSD; python ClientDaemon.py --ip_addr $CTRL_DAEMON_IP  --port $PORT" & # error

done

#!/bin/bash
#echo "1st param = $1"
#echo "2nd param = $2"
#echo "The number of params = $#"
#echo "all param : $@"
#
#DEFAULT_CTRL_DAEMON_IP="129.254.182.176"
#CTRL_DAEMON_IP=${1:-$DEFAULT_CTRL_DAEMON_IP}



python DistCtrlDaemon.py --port 2727 --map doan --target "SA 101" --num-of-learning-daemon 1 --validation-criteria 5.0
python DistCtrlDaemon.py --port 2727 --map doan --target "SA 101, SA 104" --num-of-learning-daemon 2 --validation-criteria 5.0
python DistCtrlDaemon.py --port 2727 --map doan  --target "SA 101, SA 104" --num-of-learning-daemon 2 --action gr --validation-criteria 5.0 --epoch 1 --model-save-period 1

python DistExecDaemon.py --ip_addr 129.254.182.176  --port 2727





