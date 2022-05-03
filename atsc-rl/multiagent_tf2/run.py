# -*- coding: utf-8 -*-
#
#
import argparse
import gc
import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import time


import libsalt


from config import TRAIN_CONFIG
from DebugConfiguration import DBG_OPTIONS, waitForDebug

from env.SaltEnvUtil import copyScenarioFiles
from env.SaltEnvUtil import getSaRelatedInfo
from env.SaltEnvUtil import getScenarioRelatedBeginEndTime
from env.SaltEnvUtil import makePosssibleSaNameList

from env.SappoEnv import SaltSappoEnvV3
from policy.ppoTF2 import PPOAgentTF2
from ResultCompare import compareResult

from TSOConstants import _FN_PREFIX_
from TSOUtil import addArgumentsToParser
from TSOUtil import appendLine
from TSOUtil import convertSaNameToId
from TSOUtil import findOptimalModelNum
from TSOUtil import str2bool
from TSOUtil import writeLine


def parseArgument():
    # return parseArgumentOne()
    return parseArgumentWithAddArgumentFunc()

def parseArgumentOne():
    '''
    argument parsing
    :return:
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'test', 'simulate'], default='train',
                        help='train - RL model training, test - trained model testing, simulate - fixed-time simulation before test')

    parser.add_argument('--scenario-file-path', type=str, default='data/envs/salt/',
                        help='home directory of scenario; relative path')
    parser.add_argument('--map', choices=['dj_all', 'doan', 'doan_20211207', 'sa_1_6_17'], default='sa_1_6_17',
                        help='name of map')
    # doan : SA 101, SA 104, SA 107, SA 111
    # sa_1_6_17 : SA 1,SA 6,SA 17
    parser.add_argument('--target-TL', type=str, default="SA 1,SA 6,SA 17",
                        help="target signal groups; multiple groups can be separated by comma(ex. --target-TL SA 101,SA 104)")
    parser.add_argument('--start-time', type=int, default=0, help='start time of traffic simulation; seconds')  # 25400
    parser.add_argument('--end-time', type=int, default=86400, help='end time of traffic simulation; seconds')  # 32400

    # todo hunsooni should check ddqn, ppornd, ppoea
    # parser.add_argument('--method', choices=['sappo', 'ddqn', 'ppornd', 'ppoea'], default='sappo', help='')
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

    parser.add_argument('--model-num', type=str, default='0', help='trained model number for inference')
    parser.add_argument("--result-comp", type=str2bool, default="TRUE", help='whether compare simulation result or not')

    # dockerize
    parser.add_argument('--io-home', type=str, default='.', help='home directory of io; relative path')

    ### for train
    parser.add_argument('--epoch', type=int, default=3000, help='training epoch')
    parser.add_argument('--warmup-time', type=int, default=600, help='warming-up time of simulation')
    parser.add_argument('--model-save-period', type=int, default=20, help='how often to save the trained model')
    parser.add_argument("--print-out", type=str2bool, default="TRUE", help='print result each step')

    ### action
    parser.add_argument('--action-t', type=int, default=12,
                        help='the unit time of green phase allowance')  # 녹색 신호 부여 단위 : 신호 변경 평가 주기

    ### policy : common args
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')

    ### polocy : PPO args
    parser.add_argument('--ppo-epoch', type=int, default=10, help='model fit epoch')
    parser.add_argument('--ppo-eps', type=float, default=0.1, help='')
    parser.add_argument('--_lambda', type=float, default=0.95, help='')
    parser.add_argument('--a-lr', type=float, default=0.005, help='learning rate of actor')
    parser.add_argument('--c-lr', type=float, default=0.005, help='learning rate of critic')

    # todo hunsooni should check nout used argument
    ### currently not used : logstdI, cp, mmp
    # parser.add_argument('--logstdI', type=float, default=0.5)
    #                              # currently not used : from policy/ppo.py
    # parser.add_argument('--cp', type=float, default=0.0, help='[in KC] action change penalty')
    #                             # currently not used : from env/sappo_noConst.py
    #                             # todo hunsooni  check.. SaltRewardMgmt::calculateRewardV2()
    # parser.add_argument('--mmp', type=float, default=1.0, help='min max penalty')
    #                             # currently not used

    parser.add_argument('--actionp', type=float, default=0.2,
                        help='[in KC] action 0 or 1 prob.(-1~1): Higher value_collection select more zeros')

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
    ## todo hunsooni  add 4 arguments
    ##     infer-TL : TLs to infer using trained model
    ##     infer-model-number : to indicate models which will be used to inference
    ##     infer-model-path : to specify the path that model which will be used to inference was stored
    ##     num-of-optimal-model-candidate : number of optimal model candidate
    parser.add_argument('--infer-TL', type=str, default="",
                        help="concatenate signal group with comma(ex. --infer_TL SA 101,SA 104)")

    # parser.add_argument('--infer-model-number', type=int, default=1,
    #                     help="model number which are use to discriminate the inference model")

    parser.add_argument('--infer-model-path', type=str, default=".",
                        help="directory path which are use to find the inference model")

    parser.add_argument('--num-of-optimal-model-candidate', type=int, default=3,
                        help="number of candidate to compare reward to find optimal model")

    args = parser.parse_args()

    args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.mode}.scenario.json"

    if 1:
        # todo hunsooni : think how often should we update actions
        if args.action == 'gr':
            args.control_cycle = 1

    return args



def parseArgumentWithAddArgumentFunc():
    '''
    argument parsing
    :return:
    '''

    parser = argparse.ArgumentParser()

    parser = addArgumentsToParser(parser)

    args = parser.parse_args()

    args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.mode}.scenario.json"

    if 1:
        #todo hunsooni : think how often should we update actions
        if args.action == 'gr':
            args.control_cycle = 1

    return args


def makePPOProblemVar(conf):
    '''
    make string by concatenating configuration
    this will be used as a prefix of file/path name to store log, model, ...

    :param conf:
    :return:
    '''

    problem_var = ""
    problem_var += "_state_{}".format(conf["state"])
    problem_var += "_action_{}".format(conf["action"])
    problem_var += "_reward_{}".format(conf["reward"])

    problem_var += "_gamma_{}".format(conf["gamma"])
    problem_var += "_lambda_{}".format(conf["lambda"])
    problem_var += "_alr_{}".format(conf["actor_lr"])
    problem_var += "_clr_{}".format(conf["critic_lr"])

    problem_var += "_mLen_{}".format(conf["memory_size"])
    problem_var += "_mFR_{}".format(conf["forget_ratio"])
    problem_var += "_netSz_{}".format(conf["network_layers"])
    problem_var += "_offset_range_{}".format(conf["offset_range"])
    problem_var += "_control_cycle_{}".format(conf["control_cycle"])
    # if args.method=='ppornd':
    #     problem_var += "_gammai_{}".format(args.gamma_i)
    #     problem_var += "_rndnetsize_{}".format(TRAIN_CONFIG['rnd_network_size'])
    # if args.method=='ppoea':
    #     problem_var += "_ppo_epoch_{}".format(args.ppo_epoch)
    #     problem_var += "_ppoeps_{}".format(args.ppo_eps)
    # if len(args.target_TL.split(","))==1:
    #     problem_var += "_{}".format(args.target_TL.split(",")[0])
    #
    # if args.action == 'gr' or args.action == 'gro':
    #     problem_var += "_addTime_{}".format(args.add_time)

    return problem_var


def makeDirectories(dir_name_list):
    '''
    create directories
    :param dir_name_list:
    :return:
    '''
    for dir_name in dir_name_list:
        os.makedirs(dir_name, exist_ok=True)
    return


def makePPOConfig(args):
    '''
    make configuration dictionary for PPO
    :param args: argument
    :return:
    '''

    cfg = {}

    cfg["state"] = args.state
    cfg["action"] = args.action
    cfg["reward"] = args.reward_func

    # cfg["lr"] = args.lr  # 0.005
    cfg["gamma"] = args.gamma  # 0.99
    cfg["lambda"] = args._lambda  # 0.95
    cfg["actor_lr"] = args.a_lr  # 0.005
    cfg["critic_lr"] = args.c_lr  # 0.005
    cfg["ppo_epoch"] = args.ppo_epoch  # 10
    cfg["ppo_eps"] = args.ppo_eps  # 0.1  # used for ppoea

    cfg["memory_size"] = args.mem_len
    cfg["forget_ratio"] = args.mem_fr

    cfg["offset_range"] = args.offset_range  # 2
    cfg["control_cycle"] = args.control_cycle  # 5
    cfg["add_time"] = args.add_time  # 2

    # cfg["network_layers"] = [512, 256, 128, 64, 32]  # TRAIN_CONFIG['network_size']
    cfg["network_layers"] = TRAIN_CONFIG['network_size']

    # cfg["optimizer"] = Adam
    cfg["optimizer"] = TRAIN_CONFIG['optimizer']
    return cfg



def createEnvironment(args):
    '''
    create environment
    :param args:
    :return:
    '''
    env = -1
    if args.method == 'sappo':
        env = SaltSappoEnvV3(args)
    else:
        print("internal error : {} is not supported".format(args.method))

    return env



def calculateTrialLength(args):
    '''
    calculate a length of trial using simulation start & end time (from scenario file)
    :param args:
    :return: length of trial
    '''
    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time
    return trial_len


def storeExperience(trial, step, agent, cur_state, action, reward, new_state, done, logp_t):
    '''
    store experience

    :param trial: trial
    :param step: simulation step
    :param agent:
    :param cur_state:
    :param action:
    :param reward:
    :param new_state:
    :param done:
    :param logp_t:
    :return:
    '''

    if trial == 0:
        if step == 0:
            agent.memory.reset(cur_state, action, reward, new_state, done, logp_t)
        else:
            agent.memory.store(cur_state, action, reward, new_state, done, logp_t)

    else:
        agent.memory.store(cur_state, action, reward, new_state, done, logp_t)


def makeLoadModelFnPrefix(args, problem_var):
    '''
    make a prefix of file name which indicates saved trained model parameters

    :param args:
    :param problem_var:
    :return:
    '''
    if args.infer_model_path == ".":  # default
        fn_prefix = "{}/model/{}/{}-{}-trial_{}".format(args.io_home, args.method, args.method.upper(), problem_var,
                                                        args.model_num)
    else:  # when we test distributed learning
        # /tmp/tso/SAPPO-trial_0_SA_101_actor.h5
        fn_prefix = "{}/{}-trial_{}".format(args.infer_model_path, args.method.upper(), args.model_num)

    return fn_prefix



def trainSappo(args):
    '''
    model train
      - this is work well with multiple SA
      - infer-TL is considered
    :param args:
    :return:
    '''

    ## load envs
    env = createEnvironment(args)

    ## calculate trial length using argument and scenario file
    trial_len = calculateTrialLength(args)


    ## make configuration dictionary & make some string variables
    #  : problem_var, fn_train_epoch_total_reward, fn_train_epoch_tl_reward
    if 1:

        ## make configuration dictionary
        #    and construct problem_var string to be used to create file name
        ppo_config = makePPOConfig(args)
        problem_var = makePPOProblemVar(ppo_config)

        ## construct file name to store train results(reward statistics info)
        #     : fn_train_epoch_total_reward, fn_train_epoch_tl_reward
        output_train_dir = '{}/output/train'.format(args.io_home)
        fn_train_epoch_total_reward = "{}/train_epoch_total_reward.txt".format(output_train_dir)
        fn_train_epoch_tl_reward = "{}/train_epoch_tl_reward.txt".format(output_train_dir)


    ### for tensorboard
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    train_log_dir = '{}/logs/SAPPO/{}/{}'.format(args.io_home, problem_var, time_data)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    ### 가시화 서버에서 사용할 epoch별 전체 보상 파일
    #from TSOUtil import writeLine
    writeLine(fn_train_epoch_total_reward, 'epoch,reward,40ep_reward')

    ### 가시화 서버에서 사용할 epoch별 agent별 전체 보상 파일
    writeLine(fn_train_epoch_tl_reward, 'epoch,tl_name,reward,40ep_reward')

    ep_agent_reward_list = []

    # To store reward history of each episode
    ep_reward_list = []

    # To store average reward history of last few episodes
    ma40_reward_list = []

    agent_crossName = []  # todo hunsooni should check :  currently not used
    agent_reward1, agent_reward40 = [], []

    total_reward = 0

    ## create PPO Agent
    if 1:
        agent_num = env.agent_num
        train_agent_num = env.train_agent_num
        ppo_agent = []

        for i in range(agent_num):
            target_sa = env.sa_name_list[i]
            # if 0:
            #     print("list(env.sa_obj.keys())={}".format(list(env.sa_obj.keys())))
            #     printnt("env.sa_name_list={}".format(env.sa_name_list))

            is_train_target = env.isTrainTarget(target_sa)
            ppo_config["is_train"] = is_train_target

            state_space = env.sa_obj[target_sa]['state_space']
            action_space = env.sa_obj[target_sa]['action_space']
            # # print(f"{target_sa}, state space {state_space} action space {action_space}, action min {action_min}, action max {action_max}")
            # print(f"{target_sa}, state_space={state_space}")
            # print(f"{target_sa}, action_space={action_space} action_space.shape={action_space.shape} action_space.shape[0]={action_space.shape[0]}")
            # #  SA 101, state_space=119
            # #  SA 101, action_space=Box(0, [0 0 0 4 3 5 4 3 1 1], (10,), int32)
            # #          action_space.shape=(10,)
            # #          action_space.shape[0]=10

            ##-- TF 2.x : ppo_continuous_hs,py
            action_size = action_space.shape[0]
            state_size = (state_space,)
            agent = PPOAgentTF2(env.env_name, ppo_config, action_size, state_size, target_sa.strip().replace(' ', '_'))

            if is_train_target == False:
                # make a prefix of file name which indicates saved trained model parameters
                fn_prefix = makeLoadModelFnPrefix(args, problem_var)

                if 0: # todo hunsooni should delete
                    fn_prefix = "{}/model/sappo/SAPPO-{}-trial_{}".format(args.io_home, problem_var, args.model_num)
                waitForDebug(f"agent for {target_sa} will load model parameters from {fn_prefix}")

                agent.loadModel(fn_prefix)

            ppo_agent.append(agent)

            ep_agent_reward_list.append([])
            agent_crossName.append(env.sa_obj[target_sa]['crossName_list'])  # todo hunsooni should check : currently not used

            agent_reward1.append(0)
            agent_reward40.append(0)

    ## train
    for trial in range(args.epoch):
        actions, logp_ts = [], []
        discrete_actions = []

        # 초기화
        if 1:
            for i in range(agent_num):
                target_sa = env.sa_name_list[i]
                action_space = env.sa_obj[target_sa]['action_space']
                action_size = action_space.shape[0]
                actions.append(list(0 for _ in range(action_size)))

                logp_ts.append([0])
                discrete_actions.append(list(0 for _ in range(action_size)))
                    # 고정신호의 offset을 그대로 이용하므로 0

            episodic_reward = 0
            episodic_agent_reward = [0] * agent_num
            start = time.time()

        # collect current state information
        cur_states = env.reset()

        for t in range(trial_len):
            # 새로운 action을 적용할 시기가 된것들만 모델을 이용하여 action을 만든다.
            idx_of_act_sa = env.idx_of_act_sa

            for i in idx_of_act_sa:
                observation = cur_states[i].reshape(1, -1)  # [1,2,3]  ==> [ [1,2,3] ]

                ###-- obtain actions from model
                actions[i], logp_ts[i] = ppo_agent[i].act(observation)
                actions[i], logp_ts[i] = actions[i][0], logp_ts[i][0]

                ###-- convert action : i.e., make discrete action
                sa_name = env.sa_name_list[i]
                discrete_action = env.action_mgmt.convertToDiscreteAction(sa_name, actions[i])
                discrete_actions[i] = discrete_action

            # apply all actions to env
            new_states, rewards, done, _ = env.step(discrete_actions)

            # Memorize (state, next_state, action, reward, done, logp_ts) for model training
            # 새로이 action 추론하여 적용할 리스트가 갱신되었다.
            #            이들에 대한 정보를 메모리에 저장한다.
            idx_of_act_sa = env.idx_of_act_sa

            for i in idx_of_act_sa:
                if env.sa_name_list[i] not in env.target_sa_name_list:
                    continue

                storeExperience(trial, t, ppo_agent[i], cur_states[i], actions[i], rewards[i],
                                new_states[i], done, logp_ts[i])
                # update observation
                cur_states[i] = new_states[i]
                episodic_reward += rewards[i]
                episodic_agent_reward[i] += rewards[i]

            if done:
                break

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        ma1_reward = np.mean(ep_reward_list[-1:])
        ma40_reward = np.mean(ep_reward_list[-40:])
        # print("Episode * {} * Avg Reward is ==> {} MemoryLen {}".format(trial, ma40_reward, len(state_collection[0])))
        print("Episode * {} * Avg Reward is ==> {} MemoryLen {}".format(trial, ma40_reward, ppo_agent[0].memory.getSize()))
        print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

        ### 전체 평균 보상 tensorboard에 추가
        ma40_reward_list.append(ma40_reward)


        total_reward = ma1_reward
        with train_summary_writer.as_default():
            tf.summary.scalar('train/reward', total_reward, trial)


        # do replay if it is the target of training ( this is in target SA name list )
        # the first train_agent_num agents are the target of training
        for i in range(train_agent_num):
            ppo_agent[i].replay()

            ### epoch별, 에이전트 별 평균 보상 & 40epoch 평균 보상 tensorboard에 추가
            ep_agent_reward_list[i].append(episodic_agent_reward[i])  # epoisode별 리워드 리스트에 저장

            agent_reward1[i] = np.mean(ep_agent_reward_list[i][-1:])
            agent_reward40[i] = np.mean(ep_agent_reward_list[i][-40:])

            with train_summary_writer.as_default():
                sa_name = env.target_sa_name_list[i]
                tf.summary.scalar('train_agent_reward/agent_{}'.format(sa_name), agent_reward1[i], trial)
                tf.summary.scalar('train_agent_reward_40ep_mean/agent_{}'.format(sa_name), agent_reward40[i], trial)

        train_summary_writer.flush() # update tensorboard

        ### 가시화 서버에서 사용할 epoch별 reward 파일
        appendLine(fn_train_epoch_total_reward, '{},{},{}'.format(trial, ma1_reward, ma40_reward))

        ### 가시화 서버에서 사용할 epoch별 agent별 reward 파일
        for i in range(train_agent_num):
            # the first train_agent_num agents are the target of training
            sa_name = env.sa_name_list[i]
            appendLine(fn_train_epoch_tl_reward, '{},{},{},{}'.format(trial, sa_name,
                                                                      np.mean(ep_agent_reward_list[i][-1:]),
                                                                      np.mean(ep_agent_reward_list[i][-40:])))

        ### model save
        if trial % args.model_save_period == 0:
            # fn_prefix = "{}/model/sappo/SAPPO-{}-trial_{}".format(args.io_home, problem_var, trial)
            fn_prefix = "{}/model/{}/{}-{}-trial_{}".format(args.io_home, args.method, args.method.upper(), problem_var, trial)

            for i in range(train_agent_num):
                ppo_agent[i].saveModel(fn_prefix)



        #todo hunsooni it is to handle out of memory error... I'm not sure it can handle out of memory error
        # import gc
        collected = gc.collect()

    ## find optimal model number and store it
    if DBG_OPTIONS.RunWithDistributed : # dist
        # from TSOConstants import _FN_PREFIX_
        # from TSOUtil import findOptimalModelNum
        num_of_candidate = args.num_of_optimal_model_candidate  # default 3
        model_save_period = args.model_save_period  # default 1

        # -- get the trial number that gave the best performance
        if DBG_OPTIONS.TestFindOptimalModelNum:
            optimal_model_num = findOptimalModelNum(ep_reward_list, model_save_period, num_of_candidate)
        else:
            if args.epoch == 1:
                optimal_model_num = 0
            else:
                optimal_model_num = findOptimalModelNum(ep_reward_list, model_save_period, num_of_candidate)

        # -- make the prefix of file name which stores trained model
        fn_optimal_model_prefix = "{}/model/{}/{}-{}-trial". \
            format(args.io_home, args.method, args.method.upper(), problem_var)

        # -- make the file name which stores trained model that gave the best performance
        fn_optimal_model = "{}-{}".format(fn_optimal_model_prefix, optimal_model_num)

        if DBG_OPTIONS.PrintFindOptimalModel:
            waitForDebug("run.py : return trainSappo() : fn_opt_model = {}".format(fn_optimal_model))

        # todo hunsooni 만약 여러 교차로 그룹을 대상으로 했다면 확장해야 할까? 필요없다.
        #      첫번째 그룹에 대한 정보가 전체에 대한 대표성을 가진다.
        #      이를 이용해서 학습된 모델이 저장된 경로(path) 정보와 최적 모델 번호(trial) 정보를 추출한다.
        #      실행 데몬에서 모든 target TLS에 적용하여 학습된 최적 모델을 공유 저장소에 복사한다.
        #       (ref. LearningDaemonThread::__copyTrainedModel() func )
        # fn_opt_model_info = '{}.{}'.format(_FN_PREFIX_.OPT_MODEL_INFO, args.target_TL.split(",")[0])  # strip & replace blank
        fn_opt_model_info = '{}.{}'.format(_FN_PREFIX_.OPT_MODEL_INFO, convertSaNameToId(args.target_TL.split(",")[0]))

        writeLine(fn_opt_model_info, fn_optimal_model)

        return optimal_model_num

#### test.py
def testSappo(args):

    ## load environment
    env = createEnvironment(args)

    ## calculate trial length using argument and scenario file
    trial_len = calculateTrialLength(args)


    ## make configuration dictionary & make some string variables
    if 1:
        ##-- 1. make configuration dictionary
        ppo_config = makePPOConfig(args)

        ##-- 2. construct problem_var string to be used to create file name
        problem_var = makePPOProblemVar(ppo_config)

    ## create PPO Agents & load trained model parameters
    if 1:
        agent_num = env.agent_num
        ppo_agent = []

        for i in range(agent_num):
            target_sa = env.target_sa_name_list[i]
            ppo_config["is_train"] = env.isTrainTarget(target_sa)

            state_space = env.sa_obj[target_sa]['state_space']
            action_space = env.sa_obj[target_sa]['action_space']

            action_size = action_space.shape[0]
            state_size = (state_space,)
            agent = PPOAgentTF2(env.env_name, ppo_config, action_size, state_size, convertSaNameToId(target_sa))

            # make a prefix of file name which indicates saved trained model parameters
            fn_prefix = makeLoadModelFnPrefix(args, problem_var)
            if 0:  # todo hunsooni should delete
                if args.infer_model_path == ".": # default
                    fn_prefix = "{}/model/{}/{}-{}-trial_{}".format(args.io_home, args.method, args.method.upper(), problem_var, args.model_num)
                else: # when we test distributed learning
                    fn_prefix = "{}/{}-trial_{}".format(args.infer_model_path, args.method.upper(), args.model_num)
            waitForDebug(f"agent for {target_sa} will load model parameters from {fn_prefix}")

            agent.loadModel(fn_prefix)
            ppo_agent.append(agent)




    # initialize variables which will be used to store informations when we do TEST
    if 1:
        actions, logp_ts = [], []
        discrete_actions = []
        sa_cycle = []

        for i in range(agent_num):
            target_sa = env.target_sa_name_list[i]
            sa_cycle = np.append(sa_cycle, env.sa_obj[target_sa]['cycle_list'][0])

            action_space = env.sa_obj[target_sa]['action_space']
            action_size = action_space.shape[0]
            actions.append(list(0 for _ in range(action_size)))

            logp_ts.append([0])
            discrete_actions.append(list(0 for _ in range(action_size)))
                #-- this value is zero to express fixed signal as it is

        ep_reward_list = []  # To store reward history of each episode
        episodic_reward = 0
        start = time.time()


    # collect current state information
    cur_states = env.reset()

    # do traffic simulation which are controlled by trained model(agent)
    #   1. infer & convert into action
    #   2. apply actions
    #   3. gather statistics info
    for t in range(trial_len):

        # agent들에게 현재 상태를 입력하여 출력(추론 결과)을 환경에 적용할 action으로 가공한다.
        # 1. infer by feeding current states to agents
        #   & convert inferred results into discrete actions to be applied to environment
        # do it only for the SA agents which reach time to act
        idx_of_act_sa = env.idx_of_act_sa

        for i in idx_of_act_sa:
            observation = cur_states[i].reshape(1, -1)  # [1,2,3]  ==> [ [1,2,3] ]

            # obtain actions : infer by feeding current state to agent
            actions[i], _ = ppo_agent[i].act(observation)
            actions[i] = actions[i][0]

            # convert inferred result into discrete action to be applied to environment
            sa_name = env.target_sa_name_list[i]
            discrete_action = env.action_mgmt.convertToDiscreteAction(sa_name, actions[i])
            discrete_actions[i] = discrete_action

        # 2. apply actions to environment
        new_states, rewards, done, _ = env.step(discrete_actions)

        # 3. gather statistics info
        for i in idx_of_act_sa:
            # update observation
            cur_states[i] = new_states[i]
            episodic_reward += rewards[i]

        # 4. check whether simulation is done or not
        if done:
            break

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Avg Reward is ==> {}".format(avg_reward))
    print("episode time :", time.time() - start)  # execution time =  current time - start time

    # compare traffic simulation results
    if args.result_comp:
        ft_output = pd.read_csv("{}/output/simulate/-PeriodicOutput.csv".format(args.io_home))
        rl_output = pd.read_csv("{}/output/test/-PeriodicOutput.csv".format(args.io_home))

        total_output = compareResult(args, env.tl_obj, ft_output, rl_output, args.model_num)

        result_fn = "{}/output/test/{}_{}.csv".format(args.io_home, problem_var, args.model_num)
        total_output.to_csv(result_fn, encoding='utf-8-sig', index=False)

        if 1 : # args.dist
            # todo hunsooni   Let's think about which path would be better to save it
            #                 dist learning history
            dst_fn = "{}/{}.{}.csv".format(args.infer_model_path, _FN_PREFIX_.RESULT_COMP, args.model_num)
            shutil.copy2(result_fn, dst_fn)

    return avg_reward



def fixedTimeSimulate(args):
    '''
    do traffic control with fixed signal
    :param args:
    :return:
    '''

    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    ### target tl object를 가져오기 위함
    # from env.SaltEnvUtil import makePosssibleSaNameList
    # from env.SaltEnvUtil import copyScenarioFiles
    # from env.SaltEnvUtil import getSaRelatedInfo
    salt_scenario = copyScenarioFiles(args.scenario_file_path)
    target_sa_name_list = makePosssibleSaNameList(args.target_TL)
    target_tl_obj, _, _ = getSaRelatedInfo(args, target_sa_name_list, salt_scenario)

    ### 가시화 서버용 교차로별 고정 시간 신호 기록용
    output_ft_dir = f'{args.io_home}/output/{args.mode}'
    fn_ft_phase_reward_output = f"{output_ft_dir}/ft_phase_reward_output.txt"
    writeLine(fn_ft_phase_reward_output, 'step,tl_name,actions,phase,reward')



    ### 교차로별 고정 시간 신호 기록하면서 시뮬레이션
    libsalt.start(salt_scenario)
    libsalt.setCurrentStep(start_time)
    f = open(fn_ft_phase_reward_output, mode='a+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)

    for i in range(trial_len):
        libsalt.simulationStep()
        #todo hunsooni 일정 주기로 보상 값을 얻어와서 기록한다.

        for target_tl in list(target_tl_obj.keys()):
            tlid = target_tl
            #step, tl_name, actions, phase, reward
            f.write("{},{},{},{},{}\n".format(libsalt.getCurrentStep(), target_tl_obj[target_tl]['crossName'], 0,
                                              libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid), 0))
    f.close()

    print("ft_step {}".format(libsalt.getCurrentStep()))
    libsalt.close()


def fixedTimeSimulate_new(args):
    '''
    do traffic control with fixed signal
    :param args:
    :return:
    '''

    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    ### target tl object를 가져오기 위함
    # from env.SaltEnvUtil import makePosssibleSaNameList
    # from env.SaltEnvUtil import copyScenarioFiles
    # from env.SaltEnvUtil import getSaRelatedInfo
    salt_scenario = copyScenarioFiles(args.scenario_file_path)
    target_sa_name_list = makePosssibleSaNameList(args.target_TL)
    target_tl_obj, target_sa_obj, _ = getSaRelatedInfo(args, target_sa_name_list, salt_scenario)

    ### 가시화 서버용 교차로별 고정 시간 신호 기록용
    output_ft_dir = f'{args.io_home}/output/{args.mode}'
    fn_ft_phase_reward_output = f"{output_ft_dir}/ft_phase_reward_output.txt"
    writeLine(fn_ft_phase_reward_output, 'step,tl_name,actions,phase,reward')

    if 1: # todo hunsooni should check ...reward related things
        from env.SappoRewardMgmt import  _REWARD_GATHER_UNIT_
        from env.SappoRewardMgmt import SaltRewardMgmt

        gather_unit = _REWARD_GATHER_UNIT_.SA
        num_target = len(target_sa_name_list)
        reward_mgmt = SaltRewardMgmt(args.reward_func, gather_unit, target_sa_obj, target_sa_name_list, num_target)


    ### 교차로별 고정 시간 신호 기록하면서 시뮬레이션
    libsalt.start(salt_scenario)
    libsalt.setCurrentStep(start_time)
    f = open(fn_ft_phase_reward_output, mode='a+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)

    for i in range(trial_len):
        libsalt.simulationStep()
        sim_step = libsalt.getCurrentStep()
        # todo hunsooni 일정 주기로 보상 값을 얻어와서 기록한다.

        if 1: # todo hunsooni should check ...reward related things
            sim_period = 30  # should move TRAIN_CONFIG
            reward_mgmt.gatherRewardRelatedInfo(args.action_t, sim_step, sim_period)
            for sa_idx in range(num_target):
                reward_mgmt.calculateReward(sa_idx)


        for target_tl in list(target_tl_obj.keys()):
            tlid = target_tl
            if 0:
                sa_reward = 0
            else:
                sa_name = tl_obj[tlid]['signalGroup']
                sa_idx = sa_name_list.index(sa_name)
                sa_reward = reward_mgmt.rewards[sa_idx]

            # step, tl_name, actions, phase, reward
            f.write("{},{},{},{},{}\n".format(sim_step, target_tl_obj[target_tl]['crossName'], 0,
                                              libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid), sa_reward))
    f.close()

    print("ft_step {}".format(libsalt.getCurrentStep()))
    libsalt.close()



if __name__ == "__main__":

    args = parseArgument()

    dir_name_list = [
                     f"{args.io_home}/model",
                     f"{args.io_home}/model/{args.method}",
                     f"{args.io_home}/logs",
                     f"{args.io_home}/output",
                     f"{args.io_home}/output/simulate",
                     f"{args.io_home}/output/test",
                     f"{args.io_home}/output/train",
                     f"{args.io_home}/data/envs/salt/data",
    ]
    makeDirectories(dir_name_list)

    if args.mode == 'train':
        if args.method == 'sappo':
            trainSappo(args)
        else:
            print("internal error : {} is not supported".format(args.method))

    elif args.mode == 'test':
        if args.method == 'sappo':
            testSappo(args)
        else:
            print("internal error : {} is not supported".format(args.method))

    elif args.mode == 'simulate':
        fixedTimeSimulate(args)