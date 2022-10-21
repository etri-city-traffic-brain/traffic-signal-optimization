# -*- coding: utf-8 -*-
#
#

#
#  python run.py --mode train --map doan --target "SA 101,SA 104" --action offset --epoch 2 --model-num 0 --reward-func pn --reward-gather-unit sa
#  python run.py --mode train --map doan --target "SA 101,SA 104" --action offset   --reward-func pn --reward-gather-unit sa   --model-save-period 10  --epoch 1000
#
import argparse
import datetime
import gc
import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf
import time
import sys

from deprecated import deprecated

# check environment
if 'SALT_HOME' in os.environ:
    tools = os.path.join(os.environ['SALT_HOME'], 'tools')
    sys.path.append(tools)

    tools_libsalt = os.path.join(os.environ['SALT_HOME'], 'tools/libsalt')
    sys.path.append(tools_libsalt)
else:
    sys.exit("Please declare the environment variable 'SALT_HOME'")



import libsalt

from DebugConfiguration import DBG_OPTIONS, waitForDebug

from env.SaltEnvUtil import appendPhaseRewards, gatherTsoOutputInfo, initTsoOutputInfo, appendTsoOutputInfo

if DBG_OPTIONS.RichActionOutput:
    from env.SaltEnvUtil import appendTsoOutputInfoSignal

from env.SaltEnvUtil import copyScenarioFiles
from env.SaltEnvUtil import getSaRelatedInfo
from env.SaltEnvUtil import getSimulationStartStepAndEndStep
from env.SaltEnvUtil import makePosssibleSaNameList

from env.SappoEnv import SaltSappoEnvV3

from env.SappoRewardMgmt import SaltRewardMgmtV3

from policy.ppoTF2 import PPOAgentTF2
from ResultCompare import compareResult

from TSOConstants import _FN_PREFIX_, _RESULT_COMP_, _RESULT_COMPARE_SKIP_
from TSOUtil import addArgumentsToParser
from TSOUtil import appendLine
from TSOUtil import convertSaNameToId
from TSOUtil import findOptimalModelNum
from TSOUtil import makeConfigAndProblemVar
from TSOUtil import writeLine



def parseArgument():
    '''
    argument parsing
    :return:
    '''

    parser = argparse.ArgumentParser()

    parser = addArgumentsToParser(parser)

    args = parser.parse_args()

    args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.mode}.scenario.json"

    # todo : think how often should we update actions
    # if args.action == 'gr':
    #     args.control_cycle = 1

    # to use only exploitation when we do "test"
    if args.mode == 'test':
        args.epsilon = 0.0
        args.epsilon_min = 0.0

    return args



def makeDirectories(dir_name_list):
    '''
    create directories
    :param dir_name_list:
    :return:
    '''
    for dir_name in dir_name_list:
        os.makedirs(dir_name, exist_ok=True)
    return



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


@deprecated
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



@deprecated
def storeExperience2(do_reset, agent, cur_state, action, reward, new_state, done, logp_t):
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

    if do_reset:
        agent.memory.reset(cur_state, action, reward, new_state, done, logp_t)
    else:
        agent.memory.store(cur_state, action, reward, new_state, done, logp_t)



def makeLoadModelFnPrefixV1(args, problem_var, is_train_target=False):
    '''
    make a prefix of file name which indicates saved trained model parameters

    use simple name
    it should be consistent with LearningDaemonThread::__copyTrainedModelV1() at DistExecDaemon.py

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



def makeLoadModelFnPrefixV2(args, problem_var, is_train_target=False):
    '''
    make a prefix of file name which indicates saved trained model parameters

    use complicate name : contains learning env info
    it should be consistent with LearningDaemonThread::__copyTrainedModelV2() at DistExecDaemon.py

    :param args:
    :param problem_var:
    :return:
    '''
    if args.infer_model_path == ".":  # default
        fn_prefix = "{}/model/{}/{}-{}-trial_{}".format(args.io_home, args.method, args.method.upper(), problem_var,
                                                        args.model_num)
    else:  # when we test distributed learning
        # /tmp/tso/SAPPO-trial_0_SA_101_actor.h5
        fn_prefix = "{}/{}-{}-trial_{}".format(args.infer_model_path, args.method.upper(), problem_var, args.model_num)

    return fn_prefix



def makeLoadModelFnPrefixV3(args, problem_var, is_train_target=False):
    '''
    make a prefix of file name which indicates saved trained model parameters

    it should be consistent with LearningDaemonThread::__copyTrainedModel() at DistExecDaemon.py

    v3: we consider cumulative training
    :param args:
    :param problem_var:
    :return:
    '''

    fn_prefix=""

    ## get model num to load
    if args.mode=="train":
        if is_train_target: # i.e., target-TL
            if args.cumulative_training and ( int(args.infer_model_num) >= 0 ) :
                load_model_num = args.model_num
            else:
                return fn_prefix # no need to load pre-trained model
        else: # if is_train_target == False, i.e., infer-TL
            # do not care whether cumulative_training is true or not
            load_model_num = args.infer_model_num
    else: # i.e., args.mode == "test"
        load_model_num = args.model_num


    ## construct file path
    if is_train_target and args.mode=="train":
        assert args.cumulative_training == True, "internal error : it can not happen ... should have already exited from this func "
        fn_path = "{}/model/{}".format(args.io_home, args.method)
    elif args.infer_model_path == ".":
        fn_path = "{}/model/{}".format(args.io_home, args.method)
    else:
        fn_path = args.infer_model_path

    fn_prefix = "{}/{}-{}-trial_{}".format(fn_path, args.method.upper(), problem_var, load_model_num)

    return fn_prefix



def makeLoadModelFnPrefix(args, problem_var, is_train_target=False):
    '''
    make a prefix of file name which indicates saved trained model parameters

    it should be consistent with LearningDaemonThread::__copyTrainedModel() at DistExecDaemon.py

    :param args:
    :param problem_var:
    :return:
    '''
    return makeLoadModelFnPrefixV3(args, problem_var, is_train_target)



def trainSappo(args):
    '''
    model train
      - this is work well with multiple SA
      - infer-TL is considered
    :param args:
    :return:
    '''

    ## calculate trial length using argument and scenario file
    start_time, end_time = getSimulationStartStepAndEndStep(args)
    trial_len = end_time - start_time

    # set start_/end_time which will be used to train
    args.start_time = start_time
    args.end_time = end_time

    ## load envs
    env = createEnvironment(args)

    ## make configuration dictionary & make some string variables
    #  : problem_var, fn_train_epoch_total_reward, fn_train_epoch_tl_reward
    if 1:

        ## make configuration dictionary
        #    and construct problem_var string to be used to create file name
        ppo_config, problem_var = makeConfigAndProblemVar(args)


        ## construct file name to store train results(reward statistics info)
        #     : fn_train_epoch_total_reward, fn_train_epoch_tl_reward
        output_train_dir = '{}/output/train'.format(args.io_home)
        fn_train_epoch_total_reward = "{}/train_epoch_total_reward.txt".format(output_train_dir)
        fn_train_epoch_tl_reward = "{}/train_epoch_tl_reward.txt".format(output_train_dir)


    ### for tensorboard
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    train_log_dir = '{}/logs/SAPPO/{}/{}'.format(args.io_home, problem_var, time_data)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    ### reward file for each epoch to be used by the visualization server : total
    writeLine(fn_train_epoch_total_reward, 'epoch,reward,40ep_reward')

    ### reward file for each epoch to be used in the visualization server : per TL
    writeLine(fn_train_epoch_tl_reward, 'epoch,tl_name,reward,40ep_reward')

    ep_agent_reward_list = []

    # To store reward history of each episode
    ep_reward_list = []

    # To store average reward history of last few episodes
    ma40_reward_list = []

    agent_crossName = []  # todo should check :  currently not used
    agent_reward1, agent_reward40 = [], []

    total_reward = 0
    fn_replay_memory_object = ""

    ## create PPO Agent
    if 1:
        agent_num = env.agent_num
        train_agent_num = env.train_agent_num
        ppo_agent = []

        for i in range(agent_num):
            target_sa = env.sa_name_list[i]

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

            #todo should care file name ...
            #     Should we use shared storage for distributed training?   No...
            #        Training will be done at the same node while distributed training is doing
            # for CumulateReplayMemory
            # load stored replay memory if is_train_target and args.cumulative_training and (args.infer_model_num >=0)
            if is_train_target and args.cumulative_training and (int(args.infer_model_num) >= 0) :
                fn_replay_memory_object = "{}/model/{}/{}_{}.dmp".format(args.io_home, args.method,
                                                                     _FN_PREFIX_.REPLAY_MEMORY, agent.id)
                # fn_replay_memory_object = "{}/{}_{}.dmp".format(args.infer_model_path, _FN_PREFIX_.REPLAY_MEMORY, agent.id)

                agent.loadReplayMemory(fn_replay_memory_object)
                if DBG_OPTIONS.PrintTrain:
                    print(f"### loaded len(replay memory for {agent.id})={len(agent.memory.states)}")

            fn_prefix = makeLoadModelFnPrefix(args, problem_var, is_train_target)
            if len(fn_prefix) > 0:
                if DBG_OPTIONS.PrintTrain:
                    waitForDebug(f"agent for {target_sa} will load model parameters from {fn_prefix}")
                agent.loadModel(fn_prefix)
            else:
                if DBG_OPTIONS.PrintTrain:
                    waitForDebug(
                        f"agent for {target_sa} will training without loading a pre-trained model parameter")


            ppo_agent.append(agent)

            ep_agent_reward_list.append([])
            agent_crossName.append(env.sa_obj[target_sa]['crossName_list'])  # todo should check : currently not used

            agent_reward1.append(0)
            agent_reward40.append(0)

    ## train
    for trial in range(args.epoch):
        actions, logp_ts = [], []
        discrete_actions = []

        # initialization
        if 1:
            for i in range(agent_num):
                target_sa = env.sa_name_list[i]
                action_space = env.sa_obj[target_sa]['action_space']
                action_size = action_space.shape[0]
                actions.append(list(0 for _ in range(action_size)))

                logp_ts.append([0])
                discrete_actions.append(list(0 for _ in range(action_size)))
                    # zero because the offset of the fixed signal is used as it is

            episodic_reward = 0
            episodic_agent_reward = [0] * agent_num

            if DBG_OPTIONS.PrintTrain:
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

                ppo_agent[i].memory.store(cur_states[i], actions[i], rewards[i], new_states[i], done, logp_ts[i])

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

        if DBG_OPTIONS.PrintTrain:
            train_end = time.time()
            print("Episode * {} * Avg Reward is ==> {} MemoryLen {}".format(trial, ma40_reward, ppo_agent[0].memory.getSize()))
            print("episode time :", train_end - start)  # 현재시각 - 시작시간 = 실행 시간

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



        #todo  it is to handle out of memory error... I'm not sure it can handle out of memory error
        # import gc
        collected = gc.collect()

        if DBG_OPTIONS.PrintTrain:
            replay_gc_end = time.time()
            print("replay and gc time :", replay_gc_end - train_end)  # 현재시각 - 시작시간 = 실행 시간

    ## find optimal model number and store it
    if DBG_OPTIONS.RunWithDistributed : # dist
        # from TSOConstants import _FN_PREFIX_
        # from TSOUtil import findOptimalModelNum
        num_of_candidate = args.num_of_optimal_model_candidate  # default 3
        model_save_period = args.model_save_period  # default 1

        # -- get the trial number that gave the best performance
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

        # 만약 여러 교차로 그룹을 대상으로 했다면 확장해야 할까? 필요없다.
        #      첫번째 그룹에 대한 정보가 전체에 대한 대표성을 가진다.
        #      이를 이용해서 학습된 모델이 저장된 경로(path) 정보와 최적 모델 번호(trial) 정보를 추출한다.
        #      실행 데몬에서 모든 target TLS에 적용하여 학습된 최적 모델을 공유 저장소에 복사한다.
        #       (ref. LearningDaemonThread::__copyTrainedModel() func )
        fn_opt_model_info = '{}.{}'.format(_FN_PREFIX_.OPT_MODEL_INFO, convertSaNameToId(args.target_TL.split(",")[0]))

        if int(args.infer_model_num) < 0:
            writeLine(fn_opt_model_info, fn_optimal_model)
        else:
            appendLine(fn_opt_model_info, fn_optimal_model)

        # for CumulateReplayMemory
        # todo dump-replay-memory 이용하도록 정리해야 한다.
        if args.cumulative_training:
            for i in range(agent_num):
                if not ppo_agent[i].is_train : # if it is not the target of training
                    continue

                fn_replay_memory_object = "{}/model/{}/{}_{}.dmp".format(args.io_home, args.method,
                                                                        _FN_PREFIX_.REPLAY_MEMORY, ppo_agent[i].id)

                # fn_replay_memory_object = "{}/{}_{}.dmp".format(args.infer_model_path,
                #                                                         _FN_PREFIX_.REPLAY_MEMORY, ppo_agent[i].id)
                ppo_agent[i].dumpReplayMemory(fn_replay_memory_object)
                # print(f"### dumped len(replay memory for  {ppo_agent[i].id})={len(ppo_agent[i].memory.states)}")
        else:
            print(f"args.cumulative-training is {args.cumulative_training}")


        return optimal_model_num



def testSappo(args):
    '''
    test trained model

    :param args:
    :return:
    '''

    ## calculate trial length using argument and scenario file
    start_time, end_time = getSimulationStartStepAndEndStep(args)
    trial_len = end_time - start_time

    # set start_/end_time which will be used to test
    args.start_time = start_time
    args.end_time = end_time

    ## load environment
    env = createEnvironment(args)

    ## make configuration dictionary & construct problem_var string to be used to create file names
    ppo_config, problem_var = makeConfigAndProblemVar(args)

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
            fn_prefix = makeLoadModelFnPrefix(args, problem_var, True)

            if DBG_OPTIONS.PrintTrain:
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
                #---- zero because the offset of the fixed signal is used as it is

        ep_reward_list = []  # To store reward history of each episode
        episodic_reward = 0

        if DBG_OPTIONS.PrintTrain:
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

            if DBG_OPTIONS.PrintState:
                print(f"DBG in testSappo() observation={observation}")

            # obtain actions : infer by feeding current state to agent
            actions[i], _ = ppo_agent[i].act(observation)
            actions[i] = actions[i][0]

            if DBG_OPTIONS.PrintAction :
                print(f"DBG in testSappo() actions_{i}={actions[i]}")

            # convert inferred result into discrete action to be applied to environment
            sa_name = env.target_sa_name_list[i]
            discrete_action = env.action_mgmt.convertToDiscreteAction(sa_name, actions[i])
            discrete_actions[i] = discrete_action

            if DBG_OPTIONS.PrintAction:
                print(f"DBG in testSappo() discrete_actions_{i}={discrete_actions[i]}")

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
    if DBG_OPTIONS.PrintTrain:
        print("Avg Reward is ==> {}".format(avg_reward))
        print("episode time :", time.time() - start)  # execution time =  current time - start time

    # compare traffic simulation results
    if args.result_comp:
        ft_output = pd.read_csv("{}/output/simulate/{}".format(args.io_home, _RESULT_COMP_.SIMULATION_OUTPUT))
        rl_output = pd.read_csv("{}/output/test/{}".format(args.io_home, _RESULT_COMP_.SIMULATION_OUTPUT))

        comp_skip = _RESULT_COMPARE_SKIP_
        result_fn = compareResultAndStore(args, env, ft_output, rl_output, problem_var, comp_skip)
        __printImprovementRate(env, result_fn, f'Skip {comp_skip} second')

        if DBG_OPTIONS.ResultCompareSkipWarmUp: # comparison excluding warm-up time
            comp_skip = args.warmup_time
            result_fn = compareResultAndStore(args, env, ft_output, rl_output, problem_var, comp_skip)
            __printImprovementRate(env, result_fn, f'Skip {comp_skip} second')

    return avg_reward


def compareResultAndStore(args, env, ft_output, rl_output, problem_var,  comp_skip):
    '''
    compare result of fxied-time-control and RL-agent-control
    and save the comparison results

    :param args:
    :param env:
    :param ft_output: result of traffic signal control by fixed-time
    :param rl_output: result of traffic signal control by RL-agent
    :param problem_var:
    :param comp_skip: time interval to exclude from result comparison
    :return:
    '''
    result_fn = "{}/output/test/{}_s{}_{}.csv".format(args.io_home, problem_var, comp_skip, args.model_num)
    dst_fn = "{}/{}_s{}.{}.csv".format(args.infer_model_path, _FN_PREFIX_.RESULT_COMP, comp_skip, args.model_num)
    total_output = compareResult(args, env.tl_obj, ft_output, rl_output, args.model_num, comp_skip)
    total_output.to_csv(result_fn, encoding='utf-8-sig', index=False)

    shutil.copy2(result_fn, dst_fn)

    return result_fn


def __printImprovementRate(env, result_fn, msg="Skip one hour"):
    df = pd.read_csv(result_fn, index_col=0)
    for sa in env.target_sa_name_list:
        __printImprovementRateInternal(df, sa, msg)
    __printImprovementRateInternal(df, 'total', msg)


def __printImprovementRateInternal(df, target, msg="Skip one hour"):
    ft_passed_num = df.at[target, 'ft_VehPassed_sum_0hop']
    rl_passed_num = df.at[target, 'rl_VehPassed_sum_0hop']
    ft_sum_travel_time = df.at[target, 'ft_SumTravelTime_sum_0hop']
    rl_sum_travel_time = df.at[target, 'rl_SumTravelTime_sum_0hop']

    ft_avg_travel_time = ft_sum_travel_time / ft_passed_num
    rl_avg_travel_time = rl_sum_travel_time / rl_passed_num
    imp_rate = (ft_avg_travel_time - rl_avg_travel_time) / ft_avg_travel_time * 100
    print(f'{msg} Average Travel Time ({target}): {imp_rate}% improved')



def fixedTimeSimulate(args):
    '''
    do traffic control with fixed signal
    :param args:
    :return:
    '''

    # calculate the length of simulation step of this trial : trial_len
    start_time, end_time = getSimulationStartStepAndEndStep(args)
    trial_len = end_time - start_time

    # set start_/end_time which will be used to simulate
    args.start_time = start_time
    args.end_time = end_time


    salt_scenario = copyScenarioFiles(args.scenario_file_path)
    possible_sa_name_list = makePosssibleSaNameList(args.target_TL)
    target_tl_obj, target_sa_obj, _ = getSaRelatedInfo(args, possible_sa_name_list, salt_scenario)
    target_sa_name_list = list(target_sa_obj.keys())
    target_tl_id_list = list(target_tl_obj.keys())


    ### 가시화 서버용 교차로별 고정 시간 신호 기록용
    output_ft_dir = f'{args.io_home}/output/{args.mode}'
    fn_ft_phase_reward_output = f"{output_ft_dir}/ft_phase_reward_output.txt"

    writeLine(fn_ft_phase_reward_output, 'step,tl_name,actions,phase,reward,avg_speed,avg_travel_time,sum_passed,sum_travel_time')

    reward_mgmt = SaltRewardMgmtV3(args.reward_func, args.reward_gather_unit, args.action_t,
                                       args.reward_info_collection_cycle, target_sa_obj, target_tl_obj,
                                       target_sa_name_list, len(target_sa_name_list))


    ### 교차로별 고정 시간 신호 기록하면서 시뮬레이션
    libsalt.start(salt_scenario)
    libsalt.setCurrentStep(start_time)

    actions = []

    sim_step = libsalt.getCurrentStep()

    tso_output_info_dic = initTsoOutputInfo()

    for tlid in target_tl_id_list:
        avg_speed, avg_tt, sum_passed, sum_travel_time = gatherTsoOutputInfo(tlid, target_tl_obj, num_hop=0)

        if DBG_OPTIONS.RichActionOutput:
            #todo should consider the possibility that TOD can be changed
            offset = target_tl_obj[tlid]['offset']
            duration = target_tl_obj[tlid]['duration']

            if DBG_OPTIONS.PrintAction:
                cross_name = target_tl_obj[tlid]['crossName']
                green_idx = target_tl_obj[tlid]['green_idx']
                print(f'cross_name={cross_name} offset={offset} duration={duration} green_idx={green_idx}  green_idx[0]={green_idx[0]}')
                    # cross_name=진터네거리 offset=144 duration=[18, 4, 72, 4, 18, 4, 28, 4, 25, 3] green_idx=(array([0, 2, 4, 6, 8]),)  green_idx[0]=[0 2 4 6 8]

            appendTsoOutputInfoSignal(tso_output_info_dic, offset, duration)
        tso_output_info_dic = appendTsoOutputInfo(tso_output_info_dic, avg_speed, avg_tt, sum_passed, sum_travel_time)

    for i in range(trial_len):
        libsalt.simulationStep()
        sim_step += 1

        # todo 일정 주기로 보상 값을 얻어와서 기록한다.
        appendPhaseRewards(fn_ft_phase_reward_output, sim_step, actions, reward_mgmt,
                               target_sa_obj, target_sa_name_list, target_tl_obj, target_tl_id_list,
                               tso_output_info_dic)


    print("{}... ft_step {}".format(fixedTimeSimulate.__name__, libsalt.getCurrentStep()))

    for k in tso_output_info_dic:
        tso_output_info_dic[k].clear()
    del tso_output_info_dic

    libsalt.close()



if __name__ == "__main__":

    ## dump launched time
    launched = datetime.datetime.now()
    print(f'TSO(pid={os.getpid()}) launched at {launched}')

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


    ## dump terminated time
    terminated = datetime.datetime.now()
    print(f'terminated at {terminated}')

    ## calculate & dump duration
    interval = terminated-launched
    print(f'Time taken for experiment was {interval.seconds} seconds')