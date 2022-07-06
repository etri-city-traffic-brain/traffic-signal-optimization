# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import pickle
import subprocess


from DebugConfiguration import DBG_OPTIONS
from TSOConstants import _MODE_
from TSOConstants import _MSG_TYPE_


from policy.ppoTF2 import makePPOConfig, makePPOProblemVar

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
    data = f.readlines()
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
    '''
    convert string to boolean
    :param v:
    :return:
    '''
    # import argparse
    if isinstance(v, bool):
        return v

    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def strToIntTuple(v):
    '''
    convert comma separated integer string to list of integer
    :param v: comma separated integer
    :return: list of integer
    '''
    tokens = v.split(',')
    ret_val = []
    for i in range(len(tokens)):
        try:
            ret_val.append(int(tokens[i]))
        except ValueError:
            raise argparse.ArgumentTypeError('string of comma separated integer values are expected.')

    return tuple(ret_val)



def addArgumentsToParser(parser):

    parser.add_argument('--mode', choices=['train', 'test', 'simulate'], default='train',
                        help='train - RL model training, test - trained model testing, simulate - fixed-time simulation before test')

    parser.add_argument('--scenario-file-path', type=str, default='data/envs/salt/', help='home directory of scenario; relative path')
    parser.add_argument('--map', choices=['dj_all', 'doan', 'sa_1_6_17'], default='sa_1_6_17',
                        help='name of map')
                # doan : SA 101, SA 104, SA 107, SA 111
                # sa_1_6_17 : SA 1,SA 6,SA 17
    parser.add_argument('--target-TL', type=str, default="SA 1,SA 6,SA 17",
                        help="target signal groups; multiple groups can be separated by comma(ex. --target-TL SA 101,SA 104)")
    parser.add_argument('--start-time', type=int, default=0, help='start time of traffic simulation; seconds') # 25400
    parser.add_argument('--end-time', type=int, default=86400, help='end time of traffic simulation; seconds') # 32400


    # todo should check ddqn, ppornd, ppoea
    # parser.add_argument('--method', choices=['sappo', 'ddqn', 'ppornd', 'ppoea'], default='sappo', help='')
    parser.add_argument('--method', choices=['sappo'], default='sappo', help='optimizing method')
    parser.add_argument('--action', choices=['kc', 'offset', 'gr', 'gro'], default='offset',
                        help='kc - keep or change(limit phase sequence), offset - offset, gr - green ratio, gro - green ratio+offset')
    parser.add_argument('--state', choices=['v', 'd', 'vd', 'vdd'], default='vdd',
                        help='v - volume, d - density, vd - volume + density, vdd - volume / density')
    parser.add_argument('--reward-func', choices=['pn', 'wt', 'wt_max', 'wq', 'wq_median', 'wq_min', 'wq_max', 'tt', 'cwq'],
                        default='cwq',
                        help='pn - passed num, wt - wating time, wq - waiting q length, tt - travel time, cwq - cumulative waiting q length')
                        # 'wt_SBV', 'wt_SBV_max', 'wt_ABV'  : SBV - sum-based, ABV - average-based .... TOO SLOW

    parser.add_argument("--cumulative-training", type=str2bool, default="FALSE", help='whether do cumulative training based on a previously trained model parameter or not')

    parser.add_argument('--model-num', type=str, default='0', help='trained model number')

    parser.add_argument('--infer-model-num', type=str, default='-1', help='trained model number for inference; this value is valid only when infer-TL is exist')

    parser.add_argument("--result-comp", type=str2bool, default="TRUE", help='whether compare simulation result or not')


    # dockerize
    parser.add_argument('--io-home', type=str, default='.', help='home directory of io; relative path')

    ### for train
    parser.add_argument('--epoch', type=int, default=3000, help='training epoch')
    parser.add_argument('--warmup-time', type=int, default=600, help='warming-up time of simulation')
    parser.add_argument('--model-save-period', type=int, default=20, help='how often to save the trained model')
    parser.add_argument("--print-out", type=str2bool, default="TRUE", help='print result each step')

    ### action
    parser.add_argument('--action-t', type=int, default=12, help='the unit time of green phase allowance')  # 녹색 신호 부여 단위 : 신호 변경 평가 주기

    ## reward
    parser.add_argument('--reward-info-collection-cycle', type=int, default=30, help='Information collection cycle for reward calculation')  # 녹색 신호 부여 단위 : 신호 변경 평가 주기
    parser.add_argument('--reward-gather-unit', choices=['sa', 'tl', 'env'], default='sa',
                            help='sa: sub-area, tl : traffic light, env : traffic environment ')

    ### policy : common args
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')

    ### for exploration
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon for exploration')
    parser.add_argument('--epsilon-min', type=float, default=0.1, help='minimum of epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.9999, help='epsilon decay for exploration')
    # used to adjust epsilon when we do cumulative training : ref. generateCommand() at TSOUtil.py
    parser.add_argument('--epoch-exploration-decay', type=float, default=0.9995,
                        help='epsilon decay for an epoch; has meaning when we do cumulative training')

    ### polocy : PPO args
    parser.add_argument('--ppo-epoch', type=int, default=10, help='model fit epoch')
    parser.add_argument('--ppo-eps', type=float, default=0.1, help='')
    parser.add_argument('--_lambda', type=float, default=0.95, help='')
    parser.add_argument('--a-lr', type=float, default=0.005, help='learning rate of actor')
    parser.add_argument('--c-lr', type=float, default=0.005, help='learning rate of critic')

    parser.add_argument('--network-size', type=strToIntTuple, default=(1024, 512, 512, 512, 512),
                        help='size of network in ML model; string of comma separated integer values are expected')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer for ML model')

    # todo should check nout used argument
    ### currently not used : logstdI, cp, mmp
    # parser.add_argument('--logstdI', type=float, default=0.5)
    #                              # currently not used : from policy/ppo.py
    # parser.add_argument('--cp', type=float, default=0.0, help='[in KC] action change penalty')
    #                             # currently not used : from env/sappo_noConst.py
    #                             # todo  check.. SaltRewardMgmt::calculateRewardV2()
    # parser.add_argument('--mmp', type=float, default=1.0, help='min max penalty')
    #                             # currently not used

    parser.add_argument('--actionp', type=float, default=0.2, help='[in KC] action 0 or 1 prob.(-1~1): Higher value_collection select more zeros')

    ### PPO Replay Memory
    parser.add_argument('--mem-len', type=int, default=1000, help='memory length')
    parser.add_argument('--mem-fr', type=float, default=0.9, help='memory forget ratio')

    ### SAPPO OFFSET
    parser.add_argument('--offset-range', type=int, default=2, help="offset side range")
    parser.add_argument('--control-cycle', type=int, default=5, help='')

    ### GREEN RATIO args
    parser.add_argument('--add-time', type=int, default=2, help='')

    ### currently not used : [for DDQN] replay-size, batch-size, tau, lr-update-period, lr-update-decay
    # parser.add_argument('--replay-size', type=int, default=2000) # dqn replay memory size
    # parser.add_argument('--batch-size', type=int, default=32)    # sampling size for model (batch) update
    # parser.add_argument('--tau', type=float, default=0.1)        # dqn model update ratio
    # parser.add_argument('--lr-update-period', type=int, default=5)
    # parser.add_argument('--lr-update-decay', type=float, default=0.9) # dqn : lr update decay

    ### currently not used : [for PPO RND] gamma-i
    # parser.add_argument('--gamma-i', type=float, default=0.11)

    ### currently not used : PPO + RESNET
    # parser.add_argument("--res", type=str2bool, default="TRUE")

    # --------- begin of addition
    ## add 3 arguments for distributed learning
    ##     infer-TL : TLs to infer using trained model
    ##     infer-model-path : to specify the path that model which will be used to inference was stored
    ##     num-of-optimal-model-candidate : number of optimal model candidate
    parser.add_argument('--infer-TL', type=str, default="",
                        help="concatenate signal group with comma(ex. --infer_TL SA 101,SA 104)")

    parser.add_argument('--infer-model-path', type=str, default=".",
                        help="directory path which are use to find the inference model")

    parser.add_argument('--num-of-optimal-model-candidate', type=int, default=3,
                        help="number of candidate to compare reward to find optimal model")

    return parser


'''
convert : SA name --> SA id
'''
def convertSaNameToId(in_sa_name):
    '''
    convert SA name to SA id
    removes any leading (spaces at the beginning) and trailing (spaces at the end) characters
    and replaces space with underscore(_)
    :param in_sa_name:
    :return:
    '''
    return in_sa_name.strip().replace(' ', '_')



##
##
## methods for distributed training
##
##

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
    # from TSOConstants import _MSG_TYPE_

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
    my_env = os.environ.copy()
    subprocess.SW_HIDE = 1

    r = subprocess.Popen(cmd, shell=True, env=my_env).wait() # success
    # r = subprocess.Popen(cmd, shell=False, env=my_env).wait() # error
    #    FileNotFoundError: [Errno 2] No such file or directory:
    #    'cd multiagent; python run.py --mode train --method sappo --target-TL "SA 101" --map doan --epoch 5'
    # r = subprocess.run(cmd, shell=True, env=my_env).wait() # error

    return r



def execTrafficSignalOptimization_Old(cmd):
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
    if 1: # work well
        my_env['PATH'] = env['PATH']
        my_env['SALT_HOME'] = env['SALT_HOME']
        # my_env['PYTHONPATH'] = env['PYTHONPATH']
    else: # work well
        my_env['PATH'] = env['PATH']
        salt_home = env['SALT_HOME']
        my_env['SALT_HOME'] = salt_home
        python_home = env.get('PYTHONHOME')
        if python_home == None:
            python_home = "."
        python_home = f'{python_home}:{salt_home}/tools:{salt_home}:/tools/libsalt'
        my_env['PYTHONPATH'] = python_home

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
    version : v3

    :param ep_reward_list: a list which has episode reward
    :param model_save_period: interval which indicates how open model was stored
    :param num_of_candidate: num of model to compare reward
    :return: episode number of optimal model
    '''

    ##
    num_ep = len(ep_reward_list)
    sz_slice = model_save_period * num_of_candidate
    opt_model_hint = 0

    if num_ep > sz_slice:
        loop_limit = num_ep - sz_slice

        ## 1. find some candidate of optimal model
        ##-- 1.2 initialize variables
        max_mean_reward = np.min(ep_reward_list)

        ##-- 1.2 find the range whose mean reward is maximum
        for i in range(loop_limit + 1):
            current_slice = ep_reward_list[i:i + sz_slice]
            current_slice_mean_reward = np.mean(current_slice)
            if max_mean_reward < current_slice_mean_reward:
                opt_model_hint = i # start idx of range
                max_mean_reward = current_slice_mean_reward
            if DBG_OPTIONS.PrintFindOptimalModel:
                print("i={} current_mean({})={} opt_model_hint={} max_mean_reward={}".
                      format(i, current_slice, np.mean(current_slice), opt_model_hint, current_slice_mean_reward))
        if DBG_OPTIONS.PrintFindOptimalModel:
            print("num_ep={} opt_model_hint={} ... i.e., optimal model is in range({}, {})".
                  format(num_ep, opt_model_hint, opt_model_hint, opt_model_hint + sz_slice))
    else:
        if model_save_period > 1:
            num_of_candidate = int((num_ep-1)/model_save_period) + 1
        else:
            num_of_candidate = num_ep

        if DBG_OPTIONS.PrintFindOptimalModel:
            print(f"too samll ....num_ep={num_ep} model_save_period={model_save_period} num_of_candidate={num_of_candidate}  ep_reward_list={ep_reward_list}")

    ## 2. decide which one is optimal
    ## Here we know that episode number of optimal model is in range (opt_model_hint, opt_model_hint+sz_slice)
    ##-- 2.1 calculate the first epsoide number which indicates stored model
    first_candidate = int(np.ceil(opt_model_hint / model_save_period) * model_save_period)

    # print(f"num_ep={num_ep} num_of_candidate={num_of_candidate} first_candidate={first_candidate}  ep_reward_list={ep_reward_list}")

    if DBG_OPTIONS.PrintFindOptimalModel:
        print("num_ep={} first_candidate={}  num_of_candidate={}".format(num_ep, first_candidate, num_of_candidate))

    ##-- 2.2 initialize value using first candidate
    max_ep_reward = ep_reward_list[first_candidate]
    optimal_model_num = first_candidate
    ##-- 2.3 compare rewards to find optimal model
    for i in range(1, num_of_candidate):
        next_candidate = first_candidate + model_save_period * i

        if max_ep_reward < ep_reward_list[next_candidate]:
            optimal_model_num = next_candidate
            max_ep_reward = ep_reward_list[next_candidate]
        if DBG_OPTIONS.PrintFindOptimalModel:
            print("i={}  next_candidate={} next_cand_reward={}  optimal_model_num={} max_ep_reward={}".
                  format(i, next_candidate, ep_reward_list[next_candidate], optimal_model_num, max_ep_reward))

        # print(f"i={i} num_of_candidate={num_of_candidate} next_candidate={next_candidate}  ep_reward_list={ep_reward_list} ==> opt_model_num = {optimal_model_num}")

    if DBG_OPTIONS.PrintFindOptimalModel:
        print("ZZZZZZZZZZZZZZZ found optimal_model_num={}".format(optimal_model_num))
    return optimal_model_num



def generateCommand(args):
    '''
    generate a command for traffic signal optimization

    #todo : should check this func if arguments in run.py is changed
    #       ref. addArgumentsToParser() at TSOUtil.py

    :param args: contains various command line parameters

    :return: generated command
    '''
    cmd = ' python run.py '
    cmd = cmd + ' --mode {} '.format(args.mode)
    cmd = cmd + ' --scenario-file-path {}'.format(args.scenario_file_path)
    cmd = cmd + ' --map {} '.format(args.map)
    cmd = cmd + ' --target-TL "{}" '.format(args.target_TL)
    cmd = cmd + ' --start-time {} '.format(args.start_time)
    cmd = cmd + ' --end-time {} '.format(args.end_time)

    cmd = cmd + ' --method {} '.format(args.method)
    cmd = cmd + ' --state {} '.format(args.state)
    cmd = cmd + ' --action {} '.format(args.action)
    cmd = cmd + ' --reward-func {} '.format(args.reward_func)


    # model-num   ... below
    # result-comp ... below
    cmd = cmd + ' --io-home {}'.format(args.io_home)

    if args.mode == _MODE_.TRAIN:
        cmd = cmd + ' --epoch {} '.format(args.epoch)
    else:
        cmd = cmd + ' --epoch 1 '

    cmd = cmd + ' --warmup-time {} '.format(args.warmup_time)

    if args.mode == _MODE_.TRAIN:
        cmd = cmd + ' --model-save-period {}'.format(args.model_save_period)

    cmd = cmd + ' --print-out {}'.format(args.print_out)

    cmd = cmd + ' --action-t {}'.format(args.action_t)
    cmd = cmd + ' --reward-info-collection-cycle {}'.format(args.reward_info_collection_cycle)

    cmd = cmd + ' --reward-gather-unit {}'.format(args.reward_gather_unit)
    # if args.mode == _MODE_.TRAIN:
    #     cmd = cmd + ' --reward-gather-unit sa '
    # else:
    #     cmd = cmd + ' --reward-gather-unit tl '


    cmd = cmd + ' --gamma {}'.format(args.gamma)

    ### USE_EXPLORATION_EPSILON:
    epsilon = args.epsilon

    # adjust epsilon when we do cumulative training
    if args.cumulative_training:
        an_experiment_exploration_decay = 1 - ((1 - args.epoch_exploration_decay) * args.epoch)
        experiments_exploration_decay = np.power(an_experiment_exploration_decay, args.infer_model_number + 1)
        epsilon = epsilon * experiments_exploration_decay

    cmd = cmd + ' --epsilon {}'.format(epsilon)
    cmd = cmd + ' --epsilon-min {}'.format(args.epsilon_min)
    cmd = cmd + ' --epsilon-decay {}'.format(args.epsilon_decay)

    print(f'### exp_{args.infer_model_number + 1} epsilon={epsilon}')


    cmd = cmd + ' --ppo-epoch {}'.format(args.ppo_epoch)
    cmd = cmd + ' --ppo-eps {}'.format(args.ppo_eps)
    cmd = cmd + ' --_lambda {}'.format(args._lambda)
    cmd = cmd + ' --a-lr {}'.format(args.a_lr)
    cmd = cmd + ' --c-lr {}'.format(args.c_lr)
    cmd = cmd + ' --actionp {}'.format(args.actionp)

    cmd = cmd + ' --mem-len {}'.format(args.mem_len)
    cmd = cmd + ' --mem-fr {}'.format(args.mem_fr)
    cmd = cmd + ' --offset-range {}'.format(args.offset_range)
    cmd = cmd + ' --control-cycle {}'.format(args.control_cycle)

    cmd = cmd + ' --add-time {}'.format(args.add_time)

    # infer-TL ... below
    # infer-model-path ... below
    # num-of-optimal-model-candidate ... below

    if args.mode == _MODE_.TRAIN:
        cmd = cmd + ' --num-of-optimal-model-candidate {}'.format(args.num_of_optimal_model_candidate)

        cmd = cmd + ' --cumulative-training {} '.format(args.cumulative_training)

        if args.infer_model_number >= 0:  # we have trained model... do inference
            cmd = cmd + ' --infer-TL "{}"'.format(args.infer_TL)

            # cmd = cmd + ' --cumulative-training {} '.format(args.cumulative_training)

            if 0:  # todo 0번부터 카운트하는 것을 1번부터 하게 하면 어떻까?
                load_model_num = int((args.epoch / args.model_save_period) * args.model_save_period)
            else:
                load_model_num = int((args.epoch - 1) / args.model_save_period) * args.model_save_period

            cmd = cmd + ' --model-num {} '.format(load_model_num)
            cmd = cmd + ' --infer-model-num {} '.format(args.infer_model_number)

            ## todo  만약 trial 별로 모델 저장 경로를 달리한다면 여기서 조정해야 한다.
            cmd = cmd + ' --infer-model-path {} '.format(args.model_store_root_path)

    elif args.mode == _MODE_.TEST:
        assert args.infer_model_number >= 0, f"internal error : args.infer_model_number ({args.infer_model_number}) should greater than 0 "
        # we have trained model... do inference
        cmd = cmd + ' --model-num {} '.format(args.infer_model_number)

        ## todo 만약 trial 별로 모델 저장 경로를 달리한다면 여기서 조정해야 한다.
        cmd = cmd + ' --infer-model-path {} '.format(args.model_store_root_path)

        # to compare results
        cmd = cmd + ' --result-comp True '

    if DBG_OPTIONS.PrintGeneratedCommand:
        print("{} constructed command={}".format("\n\n", cmd))

    return cmd



def makeConfigAndProblemVar(args):
    '''
    make configuration dictionary & construct problem_var string to be used to create file names
    You have to make  func.s for each policy : makePolicyConfig(), makePolicyProblemVar()

    :param args:
    :return:
    '''
    config = []
    problem_var = []

    assert args.method == 'sappo', f"Internal error in makeConfigAndProblemVar() at TSOUtil.py : methon({args.method}) is not supported"

    if args.method == 'sappo':
        config = makePPOConfig(args)
        problem_var = makePPOProblemVar(config)

    return config, problem_var




##
#
# methods for debugging
#
##


# ref. https://code.activestate.com/recipes/577504/
def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    # from __future__ import print_function
    from sys import getsizeof, stderr
    from itertools import chain
    from collections import deque

    try:
        from reprlib import repr
    except ImportError:
        pass

    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


'''
methods to test TSO Utilities
'''

def testTotalSize():
    d = dict(a=1, b=2, c=3, d=[4, 5, 6, 7], e='a string of chars')
    print(total_size(d, verbose=True))

    deep_list = [[1, 1, 1], [2, 2, 2]]
    print(total_size(deep_list, verbose=True))


def test_findOptimalModelNum():
    '''
    test findOptimalModelNum() func
    :return:
    '''
    #ep_reward_list = [0, 1, -2, 3, -4, 5, 6, -7, 8, 9, -1, 1, 2, -3, 4, 5]
    ep_reward_list = []
    ep_reward_list.append([0])
    ep_reward_list.append([0, -1])
    ep_reward_list.append([0, 1, 2])
    ep_reward_list.append([0, -1, 2, 3])
    ep_reward_list.append([0, -1, 2, 3, -4])
    ep_reward_list.append([0, 1, -2, 3, -4, 5])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, 7])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, 7, -8])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, 7, -8, 3])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, 7, -8, 3, 5])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, 7, -8, 3, -5, 1])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, 7, -8, 3, 5, -1, 2])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, 7, -8, 3, -5, -1, 2, 5])
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, -7, -8, -3, 5, 1, -2, 5, 4])

                         # 0  1   2  3   4  5  6  7   8  9  0  1  2  3  4
    ep_reward_list.append([0, 1, -2, 3, -4, 5, 6, 7, -8, 3, 5, 1, 2, 5, 4])
    ep_reward_list.append([-51, -38, -49, -23, -11, -58, -50, -21, -33, -17, -11, -14, -15, -21, -23, -28])


    model_save_period = 2
    num_of_candidate = 5
    for i in range(len(ep_reward_list)):
        opt_model_num = findOptimalModelNum(ep_reward_list[i], model_save_period, num_of_candidate)
        print("## rewards = {} opt_model_num={}\n#\n".format(ep_reward_list[i], opt_model_num))


def test_findOptimalModelNumV2():
    '''
    test findOptimalModelNum() func
    :return:
    '''
    ep_reward_list = [0, 1, -2, 3, -4, 5, 6, -7, 8, 9, -1, 1, 2, -3, 4, 5]
    ep_reward_list = [-51, -38, -49, -23, -11, -58, -50, -21, -33, -17, -11, -14, -15, -21, -23, -28]


    model_save_period = 1
    num_of_candidate = 5
    for i in range(len(ep_reward_list)):
        in_list = ep_reward_list[:i+1]
        opt_model_num = findOptimalModelNum(in_list, model_save_period, num_of_candidate)
        print("## in_list = {} opt_model_num={}\n#\n".format(in_list, opt_model_num))


if __name__ == '__main__':

    test_findOptimalModelNumV2()

    if 0:
        # writeLine("zzzzz.1", 1) # TypeError : write() argument must be str, not int
        writeLine("zzzzz.2", "contents")
        # writeLine("zzzzz.3", 3.0)  # TypeError : write() argument must be str, not float

        line = readLine("zzzzz.2")
        print(line)

        cmd = 'cd multiagent; python run.py --mode train --method sappo --target-TL "SA 101"  --map doan  --epoch 1'
        print(cmd)
        r = execTrafficSignalOptimization(cmd)

        print("returned={}".format(r))
