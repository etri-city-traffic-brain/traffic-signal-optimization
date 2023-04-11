# -*- coding: utf-8 -*-

import datetime
import socket
import threading
import time
import os
import argparse
import numpy as np
import pandas as pd
from deprecated import deprecated

from TSOUtil import doPickling, doUnpickling, Msg
from TSOConstants import _MSG_TYPE_

from DebugConfiguration import DBG_OPTIONS, waitForDebug
from TSOConstants import _INTERVAL_
from TSOConstants import _MSG_CONTENT_
from TSOConstants import _CHECK_, _MODE_, _STATE_
from TSOConstants import _FN_PREFIX_, _RESULT_COMP_
from TSOConstants import _RESULT_COMPARE_SKIP_

from TSOUtil import addArgumentsToParser
from TSOUtil import appendLine
from TSOUtil import execTrafficSignalOptimization
from TSOUtil import generateCommand
from TSOUtil import readLine
from TSOUtil import removeWhitespaceBtnComma
from TSOUtil import str2bool
from TSOUtil import writeLine



class ServingClientThread(threading.Thread):
    '''
    serve a connected client(i.e., ExecDaemon for local optimization)
    '''
    # Override Thread's __init__ method to accept the parameters needed:
    def __init__(self, channel, details, local_target, args):
        '''
        creator

        :param channel:
        :param details:
        :param local_target:
        :param args:
        '''
        self.channel = channel
        self.details = details

        self.target = self.__convertListToCommaSeperatedString(local_target)
        self.infer_model_number = -1

        self.infer_model_root_path = args.model_store_root_path
        infer_tl_list = args.target_TL.split(",")
        for t in local_target:
            infer_tl_list.remove(t)

        if len(infer_tl_list):
            self.infer_tls = self.__convertListToCommaSeperatedString(infer_tl_list)
        else:
            self.infer_tls = ''

        self.check_termination_condition = _CHECK_.NOT_READY  # validation result

        self.learned_result = 0
        self.args = args

        if DBG_OPTIONS.MaintainServerThreadState:
            self.state = _STATE_.NOT_READY
        if DBG_OPTIONS.PrintServingThread:
            print("Serving thread for {} is launched".format(self.details[0]))

        threading.Thread.__init__(self)



    def __convertListToCommaSeperatedString(self, a_list):
        '''
        convert list to comma seperated string
        :param a_list:
        :return:
        '''
        cvted = ''
        sz = len(a_list)
        for i in range(sz - 1):
            cvted = cvted + '{},'.format(a_list[i])

        cvted = cvted + '{}'.format(a_list[sz - 1])

        return cvted



    def setInferModelNumber(self, trials):
        '''
        set the model number which will be used inference
        :param trials: model number
        :return:
        '''
        self.infer_model_number = trials



    def setTerminationCondition(self, val):
        '''
        set whether to terminate
        this thread will quit if this value is set(True)
        :param val: True or False
        :return:
        '''
        self.check_termination_condition = val



    def sendMsg(self, conn, msg_type, msg_contents):
        '''
        send a message

        :param conn:
        :param msg_type:
        :param msg_contents:
        :return:
        '''
        send_msg = Msg(msg_type, msg_contents)
        pickled_msg = doPickling(send_msg)
        conn.send(pickled_msg)
        if DBG_OPTIONS.PrintServingThread:
            print("## send_msg to {}:{} -- {}".format(self.details[0], self.details[1], send_msg.toString()))
        return pickled_msg



    def receiveMsg(self, conn):
        '''
        receive a message
        :param conn:
        :return:
        '''
        recv_msg = conn.recv(2048)
        recv_msg_obj = doUnpickling(recv_msg)

        if DBG_OPTIONS.PrintServingThread:
            print("## recv_msg from {}:{} -- {}".format(self.details[0], self.details[1], recv_msg_obj.toString()))
        return recv_msg_obj



    #todo argument에 모두 넣어서 보내는 것은 어떻까?
    #     args를  dictionary나 json으로 바꾸면 좋을 것 같다.
    def makeMsgContents(self):
        '''
        make message contents
        :return:
        '''
        msg_contents_dic = {}
        # target TLs
        msg_contents_dic[_MSG_CONTENT_.TARGET_TL] = self.target

        # model number to be used to do inference
        #   do not inference but fixed_manner if this value is less than 0
        msg_contents_dic[_MSG_CONTENT_.INFER_MODEL_NUMBER] = self.infer_model_number

        # tl list to do inference
        msg_contents_dic[_MSG_CONTENT_.INFER_TL] = self.infer_tls

        # passed argument when ctrl daemon was launched
        msg_contents_dic[_MSG_CONTENT_.CTRL_DAEMON_ARGS] = self.args

        return msg_contents_dic



    def run(self):
        '''

        :return:
        '''
        is_terminate = False
        while not is_terminate:
            recv_msg_obj = self.receiveMsg(self.channel)

            if recv_msg_obj.msg_type == _MSG_TYPE_.CONNECT_OK:
                # now... connection is established
                # send local learning request
                msg_contents = self.makeMsgContents()
                send_msg_obj = self.sendMsg(self.channel, _MSG_TYPE_.LOCAL_LEARNING_REQUEST, msg_contents)
            elif recv_msg_obj.msg_type == _MSG_TYPE_.LOCAL_LEARNING_DONE:
                # local learning was done
                # set
                self.check_termination_condition = _CHECK_.ON_GOING
                self.learned_result = recv_msg_obj.msg_contents

                # wait until validation is done
                while self.check_termination_condition == _CHECK_.ON_GOING:
                    # one of { NOT_READY, ON_GOING, FAIL, SUCCESS }
                    # this value is set by main process after checking the termination condition
                    time.sleep(2)
            elif recv_msg_obj.msg_type == _MSG_TYPE_.TERMINATE_DONE :
                # local learning daemon thread was terminated
                # so, set flag to make this serving thread terminate
                is_terminate = True
                continue

            if self.check_termination_condition == _CHECK_.SUCCESS:
                self.sendMsg(self.channel, _MSG_TYPE_.TERMINATE_REQUEST, "validation success... terminate")
            elif self.check_termination_condition == _CHECK_.FAIL:
                # todo _MSG_.LOCAL_LEARNING_REQUEST 로 통합할 수 있을 것 같다.
                msg_contents = self.makeMsgContents()
                send_msg_obj = self.sendMsg(self.channel, _MSG_TYPE_.LOCAL_RELEARNING_REQUEST, msg_contents)
                self.check_termination_condition = _CHECK_.NOT_READY

        self.channel.close()

        if DBG_OPTIONS.PrintServingThread:
            print('## Closed connection:', self.details[0])



def getArgs():
    '''
    do arguments parsing
    :return: parsed argument
    '''
    parser = argparse.ArgumentParser()

    ### for distributed learning
    parser.add_argument("--port", type=int, default=2727)
    parser.add_argument("--validation-criteria", type=float, default=5.0)
    parser.add_argument("--num-of-learning-daemon", type=int, default=3)
    parser.add_argument("--model-store-root-path", type=str, default="/tmp/tso")
    parser.add_argument("--copy-simulation-output",  type=str2bool, default=False,
                        help="whether do copy simulation output(PeriodicOutput.csv) to keep test history or not")

    ### add argument for single node learning
    parser = addArgumentsToParser(parser)

    args = parser.parse_args()

    return args



def __copySimulationOutput(args, fn_origin):
    import shutil
    _origin = f'./output/{args.mode}/{fn_origin}'
    tokens = fn_origin.split(".")
    _target = f'{args.model_store_root_path}/{tokens[0]}_{args.model_num}.{tokens[1]}'

    try:
        shutil.copy2(_origin, _target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
    else:
        if DBG_OPTIONS.PrintCtrlDaemon:
            print(f'### copy {_origin} {_target}')


def validate(args, validation_trials, fn_dist_learning_history):
    '''
    check whether optimization criterior is satified
    :param args:
    :param validation_trials: used to get current performance
    :return:
    '''


    ## make command to execute TSO program with trained models

    local_learning_epoch = args.epoch
    args.mode = _MODE_.TEST
    args.epoch = 1
    args.infer_model_number = validation_trials
    args.model_num = validation_trials
    validation_cmd = generateCommand(args)
    args.epoch = local_learning_epoch

    #waitForDebug("before exec validation .... ")

    ## execute traffic signal optimization program
    result = execTrafficSignalOptimization(validation_cmd)

    ## copy simulation output file to kepp test history
    if args.copy_simulation_output:
        # copy simulation output file
        __copySimulationOutput(args, _RESULT_COMP_.SIMULATION_OUTPUT)

        # copy phase/reward output
        __copySimulationOutput(args, _RESULT_COMP_.RL_PHASE_REWARD_OUTPUT)


    # read a file which contains the result of comparison
    # and get the improvement rate

    # case 0:  when the duration of skip for result comparition is given
    comp_skip = _RESULT_COMPARE_SKIP_
    fn_result = "{}/{}_s{}.{}.csv".format(args.model_store_root_path, _FN_PREFIX_.RESULT_COMP, comp_skip,
                                          args.model_num)
    df = pd.read_csv(fn_result, index_col=0)
    imp_rate_0 = df.at[_RESULT_COMP_.ROW_NAME, _RESULT_COMP_.COLUMN_NAME]

    if DBG_OPTIONS.ResultCompareSkipWarmUp:
        # case 1:  when the duration of skip for result comparison is warming-up time
        # zz.result_comp_s600.0.csv
        comp_skip = args.warmup_time
        fn_result = "{}/{}_s{}.{}.csv".format(args.model_store_root_path, _FN_PREFIX_.RESULT_COMP, comp_skip,
                                              args.model_num)
        df = pd.read_csv(fn_result, index_col=0)
        imp_rate_1 = df.at[_RESULT_COMP_.ROW_NAME, _RESULT_COMP_.COLUMN_NAME]

    if DBG_OPTIONS.ResultCompareSkipWarmUp:
        appendLine(fn_dist_learning_history, f"{args.model_num},{imp_rate_0}, {imp_rate_1}")
    else:
        appendLine(fn_dist_learning_history, f"{args.model_num},{imp_rate_0}")

    success = _CHECK_.SUCCESS if imp_rate_0 >= args.validation_criteria else _CHECK_.FAIL

    improvement_rate = imp_rate_0
    if DBG_OPTIONS.PrintImprovementRate:
        print("improvement_rate={} got from result comp file".format(improvement_rate))

    if DBG_OPTIONS.PrintCtrlDaemon:
        waitForDebug("after check....... improvement_rate={}  validation_criteria={} success={} ".
                     format(improvement_rate, args.validation_criteria, success))

    return success


def makePartition(target, num_part):
    '''
    split targets to optimize into num_part partitions

    :param target: target to optimize
    :param num_part: # of partitions (# of exec daemon)
    :return:
    '''
    len_target = len(target)
    x = list(np.linspace(0, len_target, num_part + 1)) #
    #  if target is "SA 1, SA 2, SA 3, SA 4, SA 5" and  num_part is 3
    # print(x)  #[0.0, 1.6666666666666667, 3.3333333333333335, 5.0]

    y = [int(i) for i in x]
    # print(y) # [0, 1, 3, 5]

    partitions = []
    prev = y[0]
    for i in y[1:]:
        part = target[prev:i]

        partitions.append(part)
        prev = i

    # print (partitions)  # [['SA 1'], [' SA 2', ' SA 3'], [' SA 4', ' SA 5']]
    return partitions


####
# python DistCtrlDaemon.py --port 2727 --num_of_learning_daemon 3 --validation_criteria 6
# python DistCtrlDaemon.py --port 2727  --target-TL "SA 101, SA 104"  --num_of_learning_daemon 2 --validation_criteria 6
if __name__ == '__main__':

    if os.environ.get("UNIQ_OPT_HOME") is None:
        os.environ["UNIQ_OPT_HOME"] = os.getcwd()

    ##
    ## argument parsing
    args = getArgs()

    if args.method != 'sappo':
        print("Error : {} is not supported. Currently we only support SAPPO.".format(args.method))
        exit(1)

    args.target_TL = removeWhitespaceBtnComma(args.target_TL)
    args.infer_TL = removeWhitespaceBtnComma(args.infer_TL)

    ## create model_store_root directory if not exist
    os.makedirs(args.model_store_root_path, exist_ok=True)

    # to save the history of distributed learning
    fn_dist_learning_history = "{}/{}".format(args.model_store_root_path, _FN_PREFIX_.DIST_LEARNING_HISTORY)
    if DBG_OPTIONS.ResultCompareSkipWarmUp:
        writeLine(fn_dist_learning_history, f'trial, improvement_rate_skip{_RESULT_COMPARE_SKIP_}, improvement_rate_skip{args.warmup_time}')
    else:
        writeLine(fn_dist_learning_history, f'trial, improvement_rate_skip{_RESULT_COMPARE_SKIP_}')


    ##
    ## Set up the server:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('', args.port))  # server.bind((socket.gethostname(), 2727))
    server.listen(5)

    if DBG_OPTIONS.PrintCtrlDaemon:
        print("[CtrlDaemon] now... socket bind & listen")

    ##
    ## invoke serving threads & establish connection
    serving_client_dic = dict()

    target_list = args.target_TL.split(",")
    group_list = makePartition(target_list, args.num_of_learning_daemon)

    for i in range(args.num_of_learning_daemon):
        channel, details = server.accept()
        target = group_list[i]

        if DBG_OPTIONS.PrintCtrlDaemon:
            print("[CtrlDaemon] accept connection from {}:{}".format(details[0], details[1]))

        sc = ServingClientThread(channel, details, target, args)

        serving_client_dic[details] = sc # details[0]="1.2.3.4" details[1]=54321

        if DBG_OPTIONS.PrintCtrlDaemon:
            print("[CtrlDaemon] Now, {} clients are connected.. total num of clients to connect={}".format(len(serving_client_dic), args.num_of_learning_daemon))

    if DBG_OPTIONS.PrintCtrlDaemon:
        print("[CtrlDaemon] serving_client_dic=", serving_client_dic)


    ##
    ## start serving threads
    for sc in serving_client_dic.values():
        sc.start()

    ##
    ## serving...
    ## check termination condition
    checked_result = _CHECK_.NOT_READY

    validation_trials = 0
    while checked_result != _CHECK_.SUCCESS:

        ## get start time of current round
        curent_round_start_time = datetime.datetime.now()

        done = 0
        num_clients = len(serving_client_dic.values())

        # wait until all connected client (learning) daemon finish their learning
        while num_clients != done :
            time.sleep(_INTERVAL_.LEARNING_DONE_CHECK)
            done = 0
            for sc in serving_client_dic.values():
                if sc.check_termination_condition == _CHECK_.ON_GOING:
                    done += 1

        ### check if re-learning is needed
        checked_result = validate(args, validation_trials, fn_dist_learning_history)

        ## get end time of current round
        curent_round_end_time = datetime.datetime.now()

        ## calculate & dump elapsed time of current round
        current_round_elapsed_time = curent_round_end_time - curent_round_start_time
        print(f'Time taken for {validation_trials}-th round experiment was {current_round_elapsed_time.seconds} seconds')


        ### set the checked result : state
        for sc in serving_client_dic.values():
            sc.setInferModelNumber(validation_trials)
            sc.setTerminationCondition(checked_result) # set checked result

        validation_trials += 1

        time.sleep(_INTERVAL_.NEXT_TRIAL)

####
# socket
#   ref. https://docs.python.org/ko/3/howto/sockets.html
#        https://docs.python.org/ko/3/library/socket.html
#
# argparser
#   ref.https://donghwa-kim.github.io/argparser.html
#
