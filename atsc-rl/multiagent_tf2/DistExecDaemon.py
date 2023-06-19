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
from TSOConstants import _LENGTH_OF_MAX_MSG_
from TSOUtil import convertSaNameToId
from TSOUtil import execTrafficSignalOptimization, generateCommand, readLine, readLines
from TSOUtil import removeWhitespaceBtnComma

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
          ex. SAPPO-_state_vdd_action_gro_reward_..._offset_range_2-trial-178_SA_104_actor.h5

        Consistency must be maintained with makeLoadModelFnPrefix() at run.py

        :return:
        '''

        target = self.target_TLG
        target_list = target.split(",")
        assert target==self.args.target_TL, \
            "internal error to manage target : not equal '{}' and '{}'".format(target, self.args.target_TL)

        if DBG_OPTIONS.PrintExecDaemon:
            waitForDebug("target={}".format(self.args.target_TL))


        model_store_path = self.args.model_store_root_path

        fn_opt_model_info = '{}.{}'.format(_FN_PREFIX_.OPT_MODEL_INFO, convertSaNameToId(target_list[0]))
            # ex.,  zz.optimal_model_info.SA_104
        opt_model_info = readLines(fn_opt_model_info)[-1]
            # ex.,  ./model/sappo/SAPPO-_state_vdd_action_gro_reward_cwq.._offset_range_2-trial-178

        tokens = opt_model_info.split('-')
        opt_model_num = int(tokens[-1])  # ex., 178
        trial = self.args.infer_model_number + 1
        method = self.args.method

        path = ('-').join(tokens[:-1])  # ./model/sappo/SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial
        problem_var = path.split('/')[-1]  # SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial
        problem_var = ('-').join(problem_var.split('-')[:-1])  # SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1

        path = path + '*'  # ./model/sappo/SAPPO-_state_vdd_action_gr_reward_cwq_...._cycle_1-trial*
        #  ex., file name of trained model : SAPPO-_state_vdd_action_gro_reward_cwq_..._offset_range_2-trial_178_SA_107_actor.h5

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
            # bugfix-20230614 : tl이  SA_3인 경우, filtered_filelist에  SA_3, SA_38, SA_301 등이 모두 포함될 수 있다.
            #filter = "-trial_{}_{}".format(opt_model_num, tl)
            filter = "-trial_{}_{}_".format(opt_model_num, tl)

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
                    tokens = fname.split(f"_{tl}_") # ["SAPPO_..._trial_0", "_SA_101_", "critic_0.h5"]
                    # tokens[-1]= actor.h5, critic_0.h5, critic_1.h5
                    cmd = 'cp "{}" "{}/{}-trial_{}_{}_{}"'.format(fname, model_store_path, problem_var,
                                                                         trial, tl, tokens[-1])
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
        #     there is a possibility that it is not work if len(msg) is greater than TSOConstants._LENGTH_OF_MAX_MSG_
        recv_msg = soc.recv(_LENGTH_OF_MAX_MSG_)
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

        # 1. prepare LDT(learning-daemon-thread)s for learning
        # (1-a) create LDT(learning-daemon-thread)s for learning if it is first trial
        # (1-b) set is_learning_done flag False if it is not first trial
        if len(self.learning_daemon_thread_dic) == 0:
            # (1-a) create LDT(learning-daemon-thread)s for learning if it is first trial
            if self.do_parallel:
                ## create multiple LDT to train in parallel : one for each group of intersections
                target_tl_list = args.target_TL.split(",")

                for tlg in target_tl_list:
                    # print(f"\nnow targetTL=[{tlg}]")
                    # -- copy an object to be used as an input argument when creating LDT
                    # ----
                    new_args = copy.deepcopy(args)
                    tlg = tlg.strip()
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

                result = self.doLocalLearning(recv_msg_obj)

                # after local learning was done, send MSG_LOCAL_LEARNING_DONE
                send_msg = self.sendMsg(soc, _MSG_TYPE_.LOCAL_LEARNING_DONE, result)

            elif recv_msg_obj.msg_type == _MSG_TYPE_.TERMINATE_REQUEST:
                # set termination mode
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
# python DistExecDaemon.py --ip-addr 129.254.182.176 --port 2727 --do-parallel true
if __name__ == '__main__':
    if os.environ.get("UNIQ_OPT_HOME") is None:
        os.environ["UNIQ_OPT_HOME"] = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2727)
    parser.add_argument("--ip-addr", type=str, default="129.254.182.176")
    parser.add_argument("--do-parallel", type=str2bool, default=True)

    args = parser.parse_args()

    exec_daemon = ExecDaemon(args.ip_addr, args.port, args.do_parallel)
    exec_daemon.serve()

