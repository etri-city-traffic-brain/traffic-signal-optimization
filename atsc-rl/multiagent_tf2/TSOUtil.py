# -*- coding: utf-8 -*-
import os
import subprocess
from DebugConfiguration import DBG_OPTIONS
from TSOConstants import _MODE_
import pickle
# The pickle module implements binary protocols
#   for serializing and de-serializing a Python object structure.


def doPickling(some_obj):
    '''
    do picking :  a Python object hierarchy is converted into a byte stream
    :param some_obj:
    :return: pickled object
    '''
    pickled_obj = pickle.dumps(some_obj)
    return pickled_obj


# unpickling
#   a byte stream (from a binary file or bytes-like object)
#   is converted back into an object hierarchy
def doUnpickling(pickled):
    '''
    a  byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy
    :param pickled: 
    :return: unpickled obj
    '''
    unpickled_obj = pickle.loads(pickled, encoding='utf-8')
    # print("unpickled type:{} value:{}".format(type(unpickled_obj), unpickled_obj))
    return unpickled_obj


class Msg:
    '''
    a class to handle message
    '''
    from TSOConstants import _MSG_TYPE_

    dic_msg_type_to_string = {
        _MSG_TYPE_.CONNECT_OK: "MSG_CONNECT_OK",
        _MSG_TYPE_.LOCAL_LEARNING_REQUEST: "MSG_LOCAL_LEARNING_REQUEST",
        _MSG_TYPE_.LOCAL_RELEARNING_REQUEST: "MSG_LOCAL_RELEARNING_REQUEST",
        _MSG_TYPE_.LOCAL_LEARNING_DONE: "MSG_LOCAL_LEARNING_DONE",
        _MSG_TYPE_.TERMINATE_REQUEST: "MSG_TERMINATE_REQUEST",
        _MSG_TYPE_.TERMINATE_DONE: "MSG_TERMINATE_DONE",
        _MSG_TYPE_.READY: "MSG_READY"
    }

    def __init__(self, msg_type, msg_contents):
        '''
        constructor
        :param msg_type:
        :param msg_contents:
        '''
        self.version = 1.0
        self.msg_type = msg_type
        self.msg_contents = msg_contents

    def toString(self):
        sb = "version={}, msg_type={}, msg_content={}".format(self.version, self.dic_msg_type_to_string[self.msg_type],
                                                              self.msg_contents)
        return sb





def execTrafficSignalOptimization(cmd):
    '''
    set the environment to do TSO(traffic signal optimization)
    and
    execute TSO program
    (as an external subprocess)

    :param cmd: command to launch TSO program
    :return:
    '''
    env = os.environ
    subprocess.SW_HIDE = 1

    my_env = {}
    my_env['PATH'] = env['PATH']
    my_env['SALT_HOME'] = env['SALT_HOME']
    my_env['PYTHONPATH'] = env['PYTHONPATH']

    r = subprocess.Popen(cmd, shell=True, env=my_env).wait() # success
    # r = subprocess.Popen(cmd, shell=False, env=my_env).wait() # error
    #    FileNotFoundError: [Errno 2] No such file or directory:
    #    'cd multiagent; python run.py --mode train --method sappo --target-TL "SA 101" --map doan --epoch 5'
    # r = subprocess.run(cmd, shell=True, env=my_env).wait() # error

    return r


def findOptimalModelNum(ep_reward_list, model_save_period, num_of_candidate):
    '''
    scan episode rewards and find an episode number of optimal model
    the found episode number should indicate stored model

    :param ep_reward_list: a list which has episode reward
    :param model_save_period: interval which indicates how open model was stored
    :param num_of_candidate: num of model to compare reward
    :return: episode number of optimal model
    '''
    import numpy as np

    ##
    num_ep = len(ep_reward_list)
    sz_slice = model_save_period * num_of_candidate
    loop_limit = num_ep - sz_slice

    ## find some candidate of optimal model
    ##-- initialize variables
    max_mean_reward = np.min(ep_reward_list)
    opt_model_hint = -1

    ##-- find the range whose mean reward is maximum
    for i in range(loop_limit + 1):
        current_slice = ep_reward_list[i:i + sz_slice]
        current_slice_mean_reward = np.mean(current_slice)
        if max_mean_reward < current_slice_mean_reward:
            opt_model_hint = i
            max_mean_reward = current_slice_mean_reward
        if DBG_OPTIONS.PrintFindOptimalModel:
            print("i={} current_mean({})={} opt_model_hint={} max_mean_reward={}".
                  format(i, current_slice, np.mean(current_slice), opt_model_hint, current_slice_mean_reward))
    if DBG_OPTIONS.PrintFindOptimalModel:
        print("opt_model_hint={} ... i.e., optimal model is in range({}, {})".
              format(opt_model_hint, opt_model_hint, opt_model_hint + sz_slice))

    ## decide which one is optimal
    ## Here we know that episode number of optimal model is in range (opt_model_hint, opt_model_hint+sz_slice)
    ##-- calculate the first epsoide number which indicates stored model
    first_candidate = int(np.ceil(opt_model_hint / model_save_period) * model_save_period)

    if DBG_OPTIONS.PrintFindOptimalModel:
        print("\n\nfirst_candidate={}".format(first_candidate))
    ##-- initialize value
    max_ep_reward = ep_reward_list[first_candidate]
    optimal_model_num = first_candidate
    ##-- compare rewards to find optimal model
    for i in range(1, num_of_candidate):
        next_candidate = first_candidate + model_save_period * (i)
        if max_ep_reward < ep_reward_list[next_candidate]:
            optimal_model_num = next_candidate
            max_ep_reward = ep_reward_list[next_candidate]
        if DBG_OPTIONS.PrintFindOptimalModel:
            print("i={}  next_candidate={} next_cand_reward={}  optimal_model_num={} max_ep_reward={}".
                  format(i, next_candidate, ep_reward_list[next_candidate], optimal_model_num, max_ep_reward))
    return optimal_model_num




def generateCommand(args):
    '''
    generate a command for traffic signal optimization

    :param args: contains various command line parameters
    :param validation_trials:

    :return: generated command
    '''
    cmd = ' python run.py '
    cmd = cmd + ' --mode {} '.format(args.mode)
    cmd = cmd + ' --scenario-file-path {}'.format(args.scenario_file_path)
    cmd = cmd + ' --map {} '.format(args.map)
    cmd = cmd + ' --target-TL "{}" '.format(args.target)
    cmd = cmd + ' --start-time "{}" '.format(args.start_time)
    cmd = cmd + ' --end-time "{}" '.format(args.end_time)

    cmd = cmd + ' --method {} '.format(args.method)
    cmd = cmd + ' --state {} '.format(args.state)
    cmd = cmd + ' --action {} '.format(args.action)
    cmd = cmd + ' --reward-func {} '.format(args.reward_func)

    cmd = cmd + ' --epoch {} '.format(args.epoch)
    cmd = cmd + ' --warmup-time {} '.format(args.warmup_time)
    cmd = cmd + ' --model-save-period {}'.format(args.model_save_period)
    cmd = cmd + ' --print-out {}'.format(args.print_out)


    if args.mode == _MODE_.TRAIN:
        cmd = cmd + ' --num-of-optimal-model-candidate {}'.format(args.num_of_optimal_model_candidate)

    if args.infer_model_number >= 0: # we have trained model... do inference
        if args.mode == _MODE_.TRAIN:
            cmd = cmd + ' --infer-TL "{}"'.format(args.infer_TL)

        if ( (args.mode == _MODE_.TEST)  or (args.mode == _MODE_.TRAIN) ):
            cmd = cmd + ' --model-num {} '.format(args.infer_model_number)

            cmd = cmd + ' --infer-model-number {} '.format(args.infer_model_number)

            ## todo hunsooni 만약 trial 별로 모델 저장 경로를 달리한다면 여기서 조정해야 한다.
            cmd = cmd + ' --infer-model-path {} '.format(args.model_store_root_path)

        if args.mode == _MODE_.TEST:
            # to compare results
            cmd = cmd + ' --result-comp True '


    if DBG_OPTIONS.PrintGeneratedCommand:
        print("{} constructed command={}".format("\n\n", cmd))

    return cmd


def convertSaNameToId(in_sa_name):
    return in_sa_name.strip().replace(' ', '_')

'''
methods for file io
'''
def appendLine(fn, contents):
    '''
    append contents to a file with a given name
    :param fn: file name
    :param contents: contents to store
    :return:
    '''
    f = open(fn, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                     newline=None, closefd=True, opener=None)
    f.write("{}\n".format(contents))
    f.close()


def readAll(fn):
    '''
    read all contents from a file with given name

    :param fn: file name to read a line
    :return: str : read data
    '''
    f = open(fn, 'r')
    data = f.read()
    f.close()
    return data


def readLine(fn):
    '''
    read a line from a file with given name

    :param fn: file name to read a line
    :return: read data
    '''
    f = open(fn, 'r')
    data = f.readline()
    f.close()
    return data


def readLines(fn):
    '''
    read all lines from a file with given name

    :param fn: file name to read a line
    :return: list: read data
    '''
    f = open(fn, 'r')
    data = f.readline()
    f.close()
    return data


def writeLine(fn, contents):
    '''
    write contents to a file with a given name
    :param fn: file name
    :param contents: contents to store
    :return:
    '''
    f = open(fn, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None, closefd=True, opener=None)
    f.write("{}\n".format(contents))
    f.close()



'''
arguemnt parsing
'''
def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v

    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




'''
methods to test TSO Utilities
'''
def test_findOptimalModelNum():
    '''
    test findOptimalModelNum() func
    :return:
    '''
    ep_reward_list = [0, 1, -2, 3, -4, 5, 6, -7, 8, 9, -1, 1, 2, -3, 4, 5]
    model_save_period = 2
    num_of_candidate = 3
    opt_model_num = findOptimalModelNum(ep_reward_list, model_save_period, num_of_candidate)
    print("opt_model_num={}".format(opt_model_num))



if __name__ == '__main__':

    # writeLine("zzzzz.1", 1) # TypeError : write() argument must be str, not int
    writeLine("zzzzz.2", "contents")
    # writeLine("zzzzz.3", 3.0)  # TypeError : write() argument must be str, not float

    line = readLine("zzzzz.2")
    print(line)

    cmd = 'cd multiagent; python run.py --mode train --method sappo --target-TL "SA 101"  --map doan  --epoch 1'
    print(cmd)
    r = execTrafficSignalOptimization(cmd)

    print("returned={}".format(r))