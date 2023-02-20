# -*- coding: utf-8 -*-
import copy
import socket
import threading
import time


import os
import argparse
import glob

from TSOUtil import doPickling, doUnpickling, Msg, str2bool
from TSOConstants import _MSG_TYPE_

from DebugConfiguration import DBG_OPTIONS, waitForDebug
from TSOConstants import _INTERVAL_
from TSOConstants import _MSG_CONTENT_
from TSOConstants import _FN_PREFIX_, _MODE_
from TSOUtil import execTrafficSignalOptimization, generateCommand, readLine, readLines
from TSOUtil import convertSaNameToId

# Here's our thread:
class LearningDaemonThread(threading.Thread):
    '''
    Daemon thread for local learning
    '''
    def __init__(self, args, target_tlg):
        # self.targetTLG=""  # target traffic light group(or target intersection group)
        # self.inferTLGs=""
        self.target_TLG = target_tlg.strip()
        self.args = args
        self.do_terminate = False
        self.is_learning_done = False
        threading.Thread.__init__(self)


    def setTerminate(self):
        self.do_terminate = True


    def __copyTrainedModel(self):
        ## __copyTrainedModelV2(self, recv_msg_obj)
        '''
        copy trained model into shared storage

        use complicate name : contains learning env info
          ex. SAPPO-_state_vdd_action_gr_reward_pn_..._control_cycle_1-trial_1_SA_104_actor.h5

        Consistency must be maintained with makeLoadModelFnPrefixV2() at run.py

        :return:
        '''

        target = self.target_TLG
        target_list = target.split(",")
        assert target==self.args.target_TL, \
            "internal error to manage target : not equal '{}' and '{}'".format(target, self.args.target_TL)

        if DBG_OPTIONS.PrintExecDaemon:
            waitForDebug("target={}".format(self.args.target_TL))


        model_store_path = self.args.model_store_root_path
        #todo 현재 대상이 하나인 경우만 고려하고 있다. 여러 개인 경우에 대해 고려해야 한다.
        fn_opt_model_info = '{}.{}'.format(_FN_PREFIX_.OPT_MODEL_INFO, convertSaNameToId(target_list[0]))
        opt_model_info = readLines(fn_opt_model_info)[-1]
            # ./model/sappo/SAPPO-_state_vdd_action_gr_reward_cwq_..._control_cycle_1-trial-0

        tokens = opt_model_info.split('-')
        opt_model_num = int(tokens[-1])
        trial = self.args.infer_model_number + 1
        method = self.args.method

        path = ('-').join(tokens[:-1])  # ./model/sappo/SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial
        problem_var = path.split('/')[-1]  # SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial
        problem_var = ('-').join(problem_var.split('-')[:-1])  # SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1

        path = path + '*'  # ./model/sappo/SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial*

        if DBG_OPTIONS.PrintExecDaemon:
            print("opt_model_num={}\npath={}\nproblem_var={}".format(opt_model_num, path, problem_var))
        else:
            print("opt_model_num={}".format(opt_model_num))

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
                    cmd = 'cp "{}" "{}/{}-trial_{}_{}_{}.{}"'.format(fname, model_store_path, problem_var,
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
        while not self.do_terminate :
            time.sleep(_INTERVAL_.LEARNING_DONE_CHECK)
            if self.is_learning_done == False:
                cmd = generateCommand(self.args)
                if DBG_OPTIONS.PrintExecDaemon:
                    waitForDebug("before exec optimizer .... in LDT")
                result = execTrafficSignalOptimization(cmd)
                result = self.__copyTrainedModel()
                self.is_learning_done = True
                if DBG_OPTIONS.PrintExecDaemon:
                    waitForDebug(
                        f"LDT for {self.target_TLG}  inferTL={self.args.infer_TL} infer_model_number={self.args.infer_model_number} deactivated")



class ExecDaemon:
    '''
    Daemon for local learning
    '''
    def __init__(self, ip_addr, port, do_parallel):
        self.connect_ip_addr = ip_addr  # ip address of Control-Daemon to connect
        self.connect_port = port    # port of Control-Daemon to connect
        self.do_parallel = do_parallel # whether do parallel or not
        # to save learning-daemon-thread objects
        self.learning_daemon_thread_dic = dict() # targetTLG --> instance(LearningDaemonThread)

        if DBG_OPTIONS.PrintExecDaemon:
            print("dbg : exec daemon to connect {}:{} was created".format(ip_addr, port))


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
        #todo check the length of msg...
        #     there is a possiblity that it is not work if len(msg) is greater than 2048
        recv_msg = soc.recv(2048)
        recv_msg_obj = doUnpickling(recv_msg)
        if DBG_OPTIONS.PrintExecDaemon:
            print("## recv_msg from {}:{} -- {}".format(self.connect_ip_addr, self.connect_port, recv_msg_obj.toString()))
        return recv_msg_obj

    def doLocalLearning(self, recv_msg_obj):
        # return self.doLocalLearning_org(recv_msg_obj)
        return self.doLocalLearning_new(recv_msg_obj)

    def doLocalLearning_org(self, recv_msg_obj):
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

        # 1. prepare LDT(learning-daemon-thread)s for learning
        # (1-a) create LDT(learning-daemon-thread)s for learning if it is first trial
        # (1-b) set is_learning_done flag False if it is not first trial
        if len(self.learning_daemon_thread_dic) == 0:
            # (1-a) create LDT(learning-daemon-thread)s for learning if it is first trial
            target_tl_list = args.target_TL.split(",")

            len_target_tl_list = len(target_tl_list)
            separtor = ','
            for tlg in target_tl_list:
                # todo --> done 하나의 TLG 만 담당하게 하므로 나머지는 추론을 이용할 수 있도록 args를 조작해야 한다.
                #-- copy an object to be used as an input argument when creating LDT
                #----
                new_args = copy.deepcopy(args)
                tlg = tlg.strip()  ## todo DELETE .... remove white space
                new_args.target_TL = tlg

                remains = ""
                magic = 0
                for idx, val in enumerate(target_tl_list):
                    if val==tlg:
                        magic = 1
                        continue
                    remains += val + ('' if idx == (len_target_tl_list - 2 + magic) else separtor)

                if len(args.infer_TL.strip()) > 0 and len(remains) > 0 :
                    new_args.infer_TL=f"{args.infer_TL}, {remains}"
                elif len(args.infer_TL.strip()) > 0 and len(remains) == 0 :
                    new_args.infer_TL=f"{args.infer_TL}"
                elif len(args.infer_TL.strip()) == 0 and len(remains) == 0 :
                    new_args.infer_TL = ""
                else:
                    print("internal error ExecDaemon::doLocalLearning()")

                #-- create LDT
                ldt = LearningDaemonThread(new_args, tlg)  # allocate work
                self.learning_daemon_thread_dic[tlg] = ldt
                ldt.start()

                if DBG_OPTIONS.PrintExecDaemon:
                    waitForDebug(f"LDT for {tlg}  inferTL={new_args.infer_TL} infer_model_number={new_args.infer_model_number} launched")
        else :
            # (1-b) set is_learning_done flag False if it is not first trial
            for ldt in self.learning_daemon_thread_dic.values():
                ldt.args.infer_model_number = args.infer_model_number
                ldt.is_learning_done = False  # should set False after args.infer_model_number was assigned
                if DBG_OPTIONS.PrintExecDaemon:
                    waitForDebug(
                        f"LDT for {ldt.target_TLG}  inferTL={ldt.args.infer_TL} infer_model_number={ldt.args.infer_model_number} activated")

        # 2. check whether local learning is done for all TLG
        done = 0
        num_ldt = len(self.learning_daemon_thread_dic)
        while num_ldt != done:
            time.sleep(_INTERVAL_.LEARNING_DONE_CHECK)
            done = 0
            for ldt in self.learning_daemon_thread_dic.values():
                if ldt.is_learning_done :
                    done += 1

        return True


    ## todo 나중에 순차 실행과 비교하기 위해 할당된 모든 교차로에 대해 하나의 쓰레드(프로세스)가 학습을 하게 하는 코드 추가
    ##     DistExecDaemon 의 인자로 추가
    def doLocalLearning_new(self, recv_msg_obj):
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

        # 1. prepare LDT(learning-daemon-thread)s for learning
        # (1-a) create LDT(learning-daemon-thread)s for learning if it is first trial
        # (1-b) set is_learning_done flag False if it is not first trial
        if len(self.learning_daemon_thread_dic) == 0:
            # (1-a) create LDT(learning-daemon-thread)s for learning if it is first trial
            if self.do_parallel:
                ## create multiple LDT to train in parallel : one for each group of intersections
                target_tl_list = args.target_TL.split(",")
                len_target_tl_list = len(target_tl_list)
                separtor = ','

                for tlg in target_tl_list:
                    print(f"\nnow targetTL=[{tlg}]")
                    # todo --> done 하나의 TLG 만 담당하게 하므로 나머지는 추론을 이용할 수 있도록 args를 조작해야 한다.
                    # -- copy an object to be used as an input argument when creating LDT
                    # ----
                    new_args = copy.deepcopy(args)
                    tlg = tlg.strip()  ## todo DELETE .... remove white space
                    new_args.target_TL = tlg

                    remains = ""
                    for idx, val in enumerate(target_tl_list):
                        val = val.strip()
                        if val == tlg:
                            continue

                        remains += ", " + val
                        # print(f"\tremains=[{remains}]")

                    if len(args.infer_TL.strip()) > 0 and len(remains) > 0:
                        new_args.infer_TL = f"{args.infer_TL}{remains}"
                    elif len(args.infer_TL.strip()) > 0 and len(remains) == 0:
                        new_args.infer_TL = f"{args.infer_TL}"
                    elif len(args.infer_TL.strip()) == 0 and len(remains) == 0:
                        new_args.infer_TL = ""
                    else:
                        print("\tinternal error ExecDaemon::doLocalLearning()")

                    if DBG_OPTIONS.PrintExecDaemon:
                        print(
                            f"\tLDT for [{tlg}] new_args.targetTL=[{new_args.target_TL}] new_args.inferTL=[{new_args.infer_TL}]  launched")

                    #-- create LDT
                    ldt = LearningDaemonThread(new_args, tlg)  # allocate work
                    self.learning_daemon_thread_dic[tlg] = ldt
                    ldt.start()

            else:
                ## create a LDT for sequential training : one thread is responsible for the entire training
                tlg = args.target_TL
                # -- create LDT
                ldt = LearningDaemonThread(args, tlg)  # allocate work
                self.learning_daemon_thread_dic[tlg] = ldt
                ldt.start()

                if DBG_OPTIONS.PrintExecDaemon:
                    waitForDebug(
                        f"LDT for {tlg}  inferTL={args.infer_TL} infer_model_number={args.infer_model_number} launched")
        else :
            # (1-b) set is_learning_done flag False if it is not first trial
            for ldt in self.learning_daemon_thread_dic.values():
                ldt.args.infer_model_number = args.infer_model_number
                ldt.is_learning_done = False  # should set False after args.infer_model_number was assigned
                if DBG_OPTIONS.PrintExecDaemon:
                    waitForDebug(
                        f"LDT for {ldt.target_TLG}  inferTL={ldt.args.infer_TL} infer_model_number={ldt.args.infer_model_number} activated")

        # 2. check whether local learning is done for all TLG
        done = 0
        num_ldt = len(self.learning_daemon_thread_dic)
        while num_ldt != done:
            time.sleep(_INTERVAL_.LEARNING_DONE_CHECK)
            done = 0
            for ldt in self.learning_daemon_thread_dic.values():
                if ldt.is_learning_done :
                    done += 1

        return True



    def serve(self):

        # Connect to the server:
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.connect_ip_addr, self.connect_port))

        self.sendMsg(soc, _MSG_TYPE_.CONNECT_OK, "connection established...")

        is_terminate = False

        while not is_terminate:
            recv_msg_obj = self.receiveMsg(soc)

            if ( (recv_msg_obj.msg_type == _MSG_TYPE_.LOCAL_LEARNING_REQUEST) or
                 (recv_msg_obj.msg_type == _MSG_TYPE_.LOCAL_RELEARNING_REQUEST)) :

                # todo LDT 생성하여 학습을 진행한다.
                #      ## doLocalLearning
                #         1. LDT 생성 & 최적화할 교차로 그룹 할당 ... if first trial
                #             if len(self.learning_daemon_thread_dic) == 0
                #                   create & allocate work
                #             ### LDT들은 학습을 하고, 결과인 학습된 모델을 복사한다.
                #         2. 모든 LDT들이 학습을 종료했는지 확인한다.
                #         3. 모두 종료되었으면 성공(True)을 반환한다.
                result = self.doLocalLearning(recv_msg_obj)

                if 0:
                    # do local learning
                    result = self.doLocalLearning(recv_msg_obj)

                    result = self.__copyTrainedModel(recv_msg_obj)

                # after local learning was done, send MSG_LOCAL_LEARNING_DONE
                send_msg = self.sendMsg(soc, _MSG_TYPE_.LOCAL_LEARNING_DONE, result)

            elif recv_msg_obj.msg_type == _MSG_TYPE_.TERMINATE_REQUEST:
                # set termination mode
                # todo -> done LDT들에게 종료를 설정한다.   ----> done
                #              LDT들은 졸료가 설정되었으면 종료한다.  ---> done
                for ldt in self.learning_daemon_thread_dic.values():
                    ldt.setTerminate()

                is_terminate = True
                # send MSG_TERMINATE_DONE
                send_msg = self.sendMsg(soc, _MSG_TYPE_.TERMINATE_DONE, "terminate done")

            else:
                print("Internal Error : Unknown Message : {}".format(recv_msg_obj.toString()))
                break

        soc.close()



####
#
# python DistExecDaemon.py --ip-addr 129.254.182.176 --port 2727
if __name__ == '__main__':
    if os.environ.get("UNIQ_OPT_HOME") is None:
        os.environ["UNIQ_OPT_HOME"] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2727)
    parser.add_argument("--ip-addr", type=str, default="129.254.182.176")
    parser.add_argument("--do-parallel", type=str2bool, default="true")

    args = parser.parse_args()

    print(f"args.do_parallel={args.do_parallel}")

    exec_daemon = ExecDaemon(args.ip_addr, args.port, args.do_parallel)
    exec_daemon.serve()

