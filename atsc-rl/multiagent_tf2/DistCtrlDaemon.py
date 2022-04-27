# -*- coding: utf-8 -*-

import socket
import threading
import time
import os
import argparse
import pandas as pd

from TSOUtil import doPickling, doUnpickling, Msg
from TSOConstants import _MSG_TYPE_

from DebugConfiguration import DBG_OPTIONS, waitForDebug
from TSOConstants import _INTERVAL_
from TSOConstants import _MSG_CONTENT_
from TSOConstants import _CHECK_, _MODE_, _STATE_
from TSOConstants import _FN_PREFIX_, _IMPROVEMENT_COMP_

from TSOUtil import execTrafficSignalOptimization, generateCommand, readLine, writeLine, appendLine
from TSOUtil import str2bool


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
        self.target = local_target # optimization is done by connected client
        self.infer_model_number = -1

        self.infer_model_root_path = args.model_store_root_path
        infer_tl_list = args.target.split(",")
        infer_tl_list.remove(self.target)
        if len(infer_tl_list):
            self.infer_tls = self.convertListToCommaSeperatedString(infer_tl_list)
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


    def convertListToCommaSeperatedString(self, a_list):
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
        recv_msg = conn.recv(1024)
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
                send_msg_obj = self.sendMsg(self.channel, _MSG_TYPE_.LOCAL_RELEARNING_REQUEST,
                                            msg_contents)
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
    parser.add_argument("--ground-zero",  type=str2bool, default=False,  help="whether do simulation with fixed signal to get ground zero performance")
    parser.add_argument("--validation-criteria", type=float, default=5.0)
    parser.add_argument("--num-of-learning-daemon", type=int, default=3)
    # parser.add_argument("--infer_model_root_path", type=str, default="/tmp/tso")
    parser.add_argument("--model-store-root-path", type=str, default="/tmp/tso")
    parser.add_argument('--num-of-optimal-model-candidate', type=int, default=3,
                        help="number of candidate to compare reward to find optimal model")


    ### for single node learning
    parser.add_argument('--scenario-file-path', type=str, default='data/envs/salt')
    parser.add_argument('--map', choices=['dj_all', 'doan', 'doan_20211207', 'sa_1_6_17'], default='sa_1_6_17',
                        help='name of map')
                # doan : SA 101, SA 104, SA 107, SA 111
                # sa_1_6_17 : SA 1,SA 6,SA 17

    parser.add_argument("--target", type=str, default="SA 1,SA 6,SA 17")
    parser.add_argument('--start-time', type=int, default=0, help='start time of traffic simulation; seconds') # 25400
    parser.add_argument('--end-time', type=int, default=86400, help='end time of traffic simulation; seconds') # 32400

    parser.add_argument('--method', choices=['sappo'], default='sappo', help='optimizing method')
    parser.add_argument('--action', choices=['kc', 'offset', 'gr', 'gro'], default='offset',
                        help='kc - keep or change(limit phase sequence), offset - offset, gr - green ratio, gro - green ratio+offset')
    parser.add_argument('--state', choices=['v', 'd', 'vd', 'vdd'], default='vdd',
                        help='v - volume, d - density, vd - volume + density, vdd - volume / density')
    parser.add_argument('--reward-func',
                        choices=['pn', 'wt', 'wt_max', 'wq', 'wq_median', 'wq_min', 'wq_max', 'wt_SBV', 'wt_SBV_max',
                                 'wt_ABV', 'tt', 'cwq'],
                        default='cwq',
                        help='pn - passed num, wt - wating time, wq - waiting q length, tt - travel time, cwq - cumulative waiting q length, SBV - sum-based, ABV - average-based')

    ### for train
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--warmup-time', type=int, default=600, help='warming-up time of simulation')
    parser.add_argument('--model-save-period', type=int, default=5, help='how often to save the trained model')
    parser.add_argument("--print-out", type=str2bool, default="TRUE", help='print result each step')


    args = parser.parse_args()
    return args




def getTheCurrentPerformance(args):
    '''
    obtain the current performance
    we can use it to calculate the degree of improvement.

    :param args : argparse.Namespace : parsed argument

    :return: float
            current performance
    '''
    current_performance = 1.0

    ## make a command to run simulation with fixed signal
    local_learning_epoch = args.epoch
    args.mode=_MODE_.SIMULATE
    args.epoch = 1
    args.infer_model_number = -1 # do not inference
    cmd = generateCommand(args)
    args.epoch = local_learning_epoch


    waitForDebug("before launch simulation to get base performance")

    result = execTrafficSignalOptimization(cmd)

    return current_performance


def validate(args, validation_trials, fn_dist_learning_history):
    '''
    check whether optimization criterior is satified
    :param args:
    :param validation_trials: used to get current performance
    :return:
    '''

    #
    # todo 관련 코드 전체적으로 수정해야 한다.
    #    이미 결과 비교하는 것이 sappo_test() 에 포함되어 있다.
    #    시뮬레이터의 출력 파일을 분석하여 교차로 통과시간을 비교한다.
    #    def result_comp(args, ft_output, rl_output, model_num)를 이용한다.
    #
    #
    # sappo_test()에서 아래와 같이 수행하여 결과를 저장해 놓는다.
    # total_output.to_csv("{}/output/test/{}_{}.csv".format(args.io_home, problem_var, model_num),
    #                     encoding='utf-8-sig', index=False)


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
    if 0:
        # 개선율만을 저장한 파일에서 읽어온다.
        # open a file which contains improvement rate as a result of comparison
        # and get the value which indicates improvement rate
        fn_improvement_rate = "{}.{}".format(FN_IMPROVEMENT_RATE_PREFIX, args.model_num)
        improvement_rate = float(readLine(fn_improvement_rate))

        success = _CHECK_.SUCCESS if improvement_rate >= args.validation_criteria else _CHECK_.FAIL
        if DBG_OPTIONS.PrintImprovementRate:
            print("improvement_rate={} got from improvement_rate file ".format(improvement_rate))

    # 결과 비교 전체를 저장한 파일에서 읽어와서 개선율을 꺼낸다.
    if 1:
        # read a file which contains the result of comparison
        # and get the improvement rate
        # import pandas as pd
        # from TSOConstants import _FN_PREFIX_, _IMPROVEMENT_COMP_
        fn_result = "{}/{}.{}.csv".format(args.model_store_root_path, _FN_PREFIX_.RESULT_COMP, args.model_num)
        df = pd.read_csv(fn_result, index_col=0)

        if DBG_OPTIONS.PrintImprovementRate:
            print("### index={}".format(df.index.values))
            print("### column={}".format(df.columns.values))

        # todo hunsooni : should change hard-coding : "total", "imp_SumTravelTime_sum_0hop'
        improvement_rate = df.at['total', 'imp_SumTravelTime_sum_0hop']
        improvement_rate = df.at[_IMPROVEMENT_COMP_.ROW_NAME, _IMPROVEMENT_COMP_.COLUMN_NAME]
        success = _CHECK_.SUCCESS if improvement_rate >= args.validation_criteria else _CHECK_.FAIL

        appendLine(fn_dist_learning_history, f"{args.model_num},{improvement_rate}")

        if DBG_OPTIONS.PrintImprovementRate:
            print("improvement_rate={} got from result comp file".format(improvement_rate))

    waitForDebug("after check....... improvement_rate={}  validation_criteria={} success={} ".
                 format(improvement_rate, args.validation_criteria, success))

    return success

####
# python DistCtrlDaemon.py --port 2727 --num_of_learning_daemon 3 --validation_criteria 6 --ground-zero True
# python DistCtrlDaemon.py --port 2727  --target "SA 101, SA 104"  --num_of_learning_daemon 2 --validation_criteria 6
if __name__ == '__main__':

    if os.environ.get("UNIQ_OPT_HOME") is None:
        os.environ["UNIQ_OPT_HOME"] = os.getcwd()

    ##
    ## argument parsing
    args = getArgs()

    ## create model_store_root directory if not exist
    os.makedirs(args.model_store_root_path, exist_ok=True)

    # to save the history of distributed learning
    fn_dist_learning_history = "{}/{}".format(args.model_store_root_path, _FN_PREFIX_.DIST_LEARNING_HISTORY)
    writeLine(fn_dist_learning_history, "trial, improvement_rate")

    # local_learning_epoch = args.epoch
    # args.epoch = local_learning_epoch

    ## if result comparison is set,
    #    xxx_test() funcs in test.py obtain the performance of simulation
    #       using fixed-time-based signal control for result comparison
    ##  so.. we do not do it here
    if args.ground_zero :
        ## first obtain the current performance
        #     so that we can use it to calculate the degree of improvement.
        current_performance = getTheCurrentPerformance(args)

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

    group_list = args.target.split(",")

    assert len(group_list)==args.num_of_learning_daemon,\
        "command error : # of group({}) should be equal to num_of_learning_daemon({}}".format(len(group_list), args.num_of_learning_daemon)

    all_targets = group_list[:args.num_of_learning_daemon]
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
        done = 0
        num_clients = len(serving_client_dic.values())

        # wait until all connected client (learning) daemon finish their learning
        while num_clients != done :
            time.sleep(_INTERVAL_.LEARNING_DONE_CHECK)
            done = 0
            for sc in serving_client_dic.values():
                if sc.check_termination_condition == _CHECK_.ON_GOING:
                    done += 1

        # ### gather learned results
        # learned_result_dic = {}
        # for sc in serving_client_dic.values():
        #     learned_result_dic[sc.target] = sc.learned_result

        ### check if re-learning is needed
        checked_result = validate(args, validation_trials, fn_dist_learning_history)

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


######
#     python ServerDaemon.py --config_file_path config.json
#
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file_path", type=str, default="config.json")
    # args = parser.parse_args()
    # json_fn = args.config_file_path
    #
    # with open(json_fn, 'r') as file:
    #     json_string = file.read()
    #     data_dict = json.loads(json_string)
    #
    #     port = data_dict["port"]