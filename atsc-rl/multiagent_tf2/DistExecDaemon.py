# -*- coding: utf-8 -*-
import socket
import threading

import os
import argparse
import glob

from TSOUtil import doPickling, doUnpickling, Msg
from TSOConstants import _MSG_TYPE_

from DebugConfiguration import DBG_OPTIONS, waitForDebug
from TSOConstants import _MSG_CONTENT_
from TSOConstants import _FN_PREFIX_, _MODE_
from TSOUtil import execTrafficSignalOptimization, generateCommand, readLine
from TSOUtil import convertSaNameToId

# Here's our thread:
class LearningDaemonThread(threading.Thread):
    '''
    Daemon thread for local learning
    '''
    def __init__(self, ip_addr, port):
        self.connect_ip_addr = ip_addr
        self.connect_port = port
        threading.Thread.__init__(self)

        if DBG_OPTIONS.PrintExecDaemon:
            print("dbg : thread to connect {}:{} was created".format(ip_addr, port))

    def sendMsg(self, soc, msg_type, msg_contents):
        '''
        send a message
        :param soc:
        :param msg_type:
        :param msg_contents:
        :return:
        '''
        send_msg = Msg(msg_type, msg_contents)
        pickled_msg = doPickling(send_msg)
        soc.send(pickled_msg)
        if DBG_OPTIONS.PrintExecDaemon:
            print("## send_msg to {}:{} -- {}".format(self.connect_ip_addr, self.connect_port, send_msg.toString()))
        return pickled_msg

    def receiveMsg(self, soc):
        '''
        receive a message
        :param soc:
        :return:
        '''
        recv_msg = soc.recv(1024)
        recv_msg_obj = doUnpickling(recv_msg)
        if DBG_OPTIONS.PrintExecDaemon:
            print("## recv_msg from {}:{} -- {}".format(self.connect_ip_addr, self.connect_port, recv_msg_obj.toString()))
        return recv_msg_obj


    def doLocalLearning(self, recv_msg_obj):
        '''
        do local learning
        :param recv_msg_obj:
        :return:
        '''
        args = recv_msg_obj.msg_contents[_MSG_CONTENT_.CTRL_DAEMON_ARGS]
        args.mode = _MODE_.TRAIN
        args.target_TL = recv_msg_obj.msg_contents[_MSG_CONTENT_.TARGET_TL]
        args.infer_TL = recv_msg_obj.msg_contents[_MSG_CONTENT_.INFER_TL]
        args.infer_model_number = recv_msg_obj.msg_contents[_MSG_CONTENT_.INFER_MODEL_NUMBER]

        cmd = generateCommand(args)
        waitForDebug("before exec optimizer .... ")
        result = execTrafficSignalOptimization(cmd)

        return result

    def __copyTrainedModel(self, recv_msg_obj):
        '''
        copy trained model into shared storage
        Consistency must be maintained with makeLoadModelFnPrefix() at run.py

        :param recv_msg_obj:
        :return:
        '''
        return self.__copyTrainedModelV2(recv_msg_obj)

    def __copyTrainedModelV1(self, recv_msg_obj):
        '''
        copy trained model into shared storage

        use simple file name : ex. SAPPO-trial_1_SA_104_actor.h5

        Consistency must be maintained with makeLoadModelFnPrefixV1() at run.py

        :param recv_msg_obj:
        :return:
        '''
        target = recv_msg_obj.msg_contents[_MSG_CONTENT_.TARGET_TL]
        target_list = target.split(",")
        waitForDebug("target={} target_list={}".format(target, target_list))

        model_store_path = recv_msg_obj.msg_contents[_MSG_CONTENT_.CTRL_DAEMON_ARGS].model_store_root_path
        #todo 현재 대상이 하나인 경우만 고려하고 있다. 여러 개인 경우에 대해 고려해야 한다.
        fn_opt_model_info = '{}.{}'.format(_FN_PREFIX_.OPT_MODEL_INFO, convertSaNameToId(target_list[0]))
        fn = readLine(fn_opt_model_info)

        # 파일명을  TL 명으로 변경하여 복사
        # 가정 : TL명들을 알고 있다.
        # 1. TL 별로 관련 파일들을 가져온다  : SA 101-trial-0
        # 2. dot을 구분자로 분할하여 확장자를 얻어온다
        # 3. 해당 파일을 "TL명.확장자" 로 복사한다.
        opt_model_num = int(fn.split('-')[-1])
        trial = recv_msg_obj.msg_contents[_MSG_CONTENT_.CTRL_DAEMON_ARGS].infer_model_number + 1
        method = recv_msg_obj.msg_contents[_MSG_CONTENT_.CTRL_DAEMON_ARGS].method

        for tl in target_list:
            # 0. caution
            #  we use target id after removing spaces at the beginning and at the end of the string(target name)
            #                        and replace blank(space) in the middle of a string with underline(_)
            tl = tl.strip().replace(' ', '_')

            # 1. get related files
            # filter = "{}-trial-{}".format(tl, opt_model_num)
            filter = "-trial_{}_{}".format(opt_model_num, tl)

            tokens = fn.split('/')
            path = ('/').join(tokens[:-1])  # ./model/sappo
            path = path + '/*'  # ./model/sappo/*
            filelist = glob.glob(path)
            filtered_filelist = [fname for fname in filelist if filter in fname]

            if 1:
                print(f"\n\nfilered_filelist=")

                for z in range(len(filtered_filelist)):
                    print("\t", filtered_filelist[z])

                print("\n\n")

            for fname in filtered_filelist:

                assert method== 'sappo', f"Internal error in LearningDaemonThread::__copyTrainedModel() : methon({method}) is not supported"

                if method == 'sappo': # use PPOAgentTF2
                    # 2. get file extension
                    tokens = fname.split('.')
                    extension = tokens[-1]

                    model_name = tokens[-2].split('_')[-1]  # actor or critic
                    # 3. make command
                    cmd = 'pwd; cp "{}" "{}/{}-trial_{}_{}_{}.{}"'.format(fname, model_store_path, method.upper(),
                                                                          trial, tl, model_name, extension)
                else :
                    print("Internal error : LearningDaemonThread::__copyTrainedModel()")

                if DBG_OPTIONS.PrintExecDaemon:
                    waitForDebug(cmd)

                r = execTrafficSignalOptimization(cmd)

                assert r == 0, f"error in subprocess : subprocess returns {r}" # subprocess.Popen() returns 0 if success
                                # A negative value -N indicates that the child was terminated by signal N

        return True


    def __copyTrainedModelV2(self, recv_msg_obj):
        '''
        copy trained model into shared storage

        use complicate name : contains learning env info
          ex. SAPPO-_state_vdd_action_gr_reward_pn_..._control_cycle_1-trial_1_SA_104_actor.h5

        Consistency must be maintained with makeLoadModelFnPrefixV2() at run.py

        :param recv_msg_obj:
        :return:
        '''
        target = recv_msg_obj.msg_contents[_MSG_CONTENT_.TARGET_TL]
        target_list = target.split(",")
        waitForDebug("target={} target_list={}".format(target, target_list))

        model_store_path = recv_msg_obj.msg_contents[_MSG_CONTENT_.CTRL_DAEMON_ARGS].model_store_root_path
        #todo 현재 대상이 하나인 경우만 고려하고 있다. 여러 개인 경우에 대해 고려해야 한다.
        fn_opt_model_info = '{}.{}'.format(_FN_PREFIX_.OPT_MODEL_INFO, convertSaNameToId(target_list[0]))
        opt_model_info = readLine(fn_opt_model_info)
            # ./model/sappo/SAPPO-_state_vdd_action_gr_reward_cwq_..._control_cycle_1-trial-0

        tokens = opt_model_info.split('-')
        opt_model_num = int(tokens[-1])
        trial = recv_msg_obj.msg_contents[_MSG_CONTENT_.CTRL_DAEMON_ARGS].infer_model_number + 1
        method = recv_msg_obj.msg_contents[_MSG_CONTENT_.CTRL_DAEMON_ARGS].method

        path = ('-').join(tokens[:-1])  # ./model/sappo/SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial
        problem_var = path.split('/')[-1]  # SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial
        problem_var = ('-').join(problem_var.split('-')[:-1])  # SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1

        path = path + '*'  # ./model/sappo/SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial*
        print("opt_model_num={}\npath={}\nproblem_var={}".format(opt_model_num, path, problem_var))

        for tl in target_list:
            # 0. caution
            #  we use target id after removing spaces at the beginning and at the end of the string(target name)
            #                        and replace blank(space) in the middle of a string with underline(_)
            tl = tl.strip().replace(' ', '_')

            # 1. get related files
            # filter = "{}-trial-{}".format(tl, opt_model_num)
            filter = "-trial_{}_{}".format(opt_model_num, tl)

            filelist = glob.glob(path)
            filtered_filelist = [fname for fname in filelist if filter in fname]

            if DBG_OPTIONS.PrintExecDaemon:
                print(f"\n\nfilered_filelist=")
                for z in range(len(filtered_filelist)):
                    print("\t", filtered_filelist[z])
                print("\n\n")


            for fname in filtered_filelist:

                assert method== 'sappo', f"Internal error in LearningDaemonThread::__copyTrainedModel() : methon({method}) is not supported"

                if method == 'sappo': # use PPOAgentTF2
                    # 2. get file extension
                    tokens = fname.split('.')
                    extension = tokens[-1]

                    model_name = tokens[-2].split('_')[-1]  # actor or critic
                    # 3. make command
                    cmd = 'pwd; cp "{}" "{}/{}-trial_{}_{}_{}.{}"'.format(fname, model_store_path, problem_var,
                                                                          trial, tl, model_name, extension)
                else :
                    print("Internal error : LearningDaemonThread::__copyTrainedModel()")

                if DBG_OPTIONS.PrintExecDaemon:
                    waitForDebug(cmd)

                r = execTrafficSignalOptimization(cmd)

                assert r == 0, f"error in subprocess : subprocess returns {r}" # subprocess.Popen() returns 0 if success
                                # A negative value -N indicates that the child was terminated by signal N

        return True


    def run(self):

        # Connect to the server:
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.connect_ip_addr, self.connect_port))

        self.sendMsg(soc, _MSG_TYPE_.CONNECT_OK, "connection established...")

        is_terminate = False

        while not is_terminate:
            recv_msg_obj = self.receiveMsg(soc)

            if ( (recv_msg_obj.msg_type == _MSG_TYPE_.LOCAL_LEARNING_REQUEST) or
                 (recv_msg_obj.msg_type == _MSG_TYPE_.LOCAL_RELEARNING_REQUEST)) :

                # do local learning
                result = self.doLocalLearning(recv_msg_obj)
                if DBG_OPTIONS.PrintExecDaemon:
                    print("## returned trained opt model number ={}".format(result))

                result = self.__copyTrainedModel(recv_msg_obj)

                if DBG_OPTIONS.PrintExecDaemon:
                    print("## returned trained opt model number ={}".format(result))

                # after local learning was done, send MSG_LOCAL_LEARNING_DONE
                send_msg = self.sendMsg(soc, _MSG_TYPE_.LOCAL_LEARNING_DONE, result)

            elif recv_msg_obj.msg_type == _MSG_TYPE_.TERMINATE_REQUEST:
                # set termination mode
                is_terminate = True
                # send MSG_TERMINATE_DONE
                send_msg = self.sendMsg(soc, _MSG_TYPE_.TERMINATE_DONE, "terminate done")

            else:
                print("Internal Error : Unknown Message : {}".format(recv_msg_obj.toString()))
                break

        soc.close()


####
#
# python DistExecDaemon.py --ip_addr 129.254.182.176 --port 2727
if __name__ == '__main__':
    if os.environ.get("UNIQ_OPT_HOME") is None:
        os.environ["UNIQ_OPT_HOME"] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2727)
    parser.add_argument("--ip_addr", type=str, default="129.254.182.176")
    args = parser.parse_args()

    ct = LearningDaemonThread(args.ip_addr, args.port)
    ct.start()
