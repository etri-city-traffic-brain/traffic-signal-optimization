# -*- coding: utf-8 -*-
#
#

#
#  python run.py --mode train --map doan --target "SA 101,SA 104" --action offset --epoch 2 --model-num 0 --reward-func pn --reward-gather-unit sa
#  python run.py --mode train --map doan --target "SA 101,SA 104" --action offset   --reward-func pn --reward-gather-unit sa   --model-save-period 10  --epoch 1000
#
import argparse
import datetime
#import gc
import numpy as np
import os

from multiprocessing import Process, Pipe
from threading import Thread

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

import tensorflow as tf
def configure_gpu():
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

configure_gpu()

           
import pandas as pd
import shutil
#import tensorflow as tf
import time
import sys
import copy 
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

from env.off_ppo.SaltEnvUtil import appendPhaseRewards, gatherTsoOutputInfo, initTsoOutputInfo, appendTsoOutputInfo

if DBG_OPTIONS.RichActionOutput:
    from env.off_ppo.SaltEnvUtil import appendTsoOutputInfoSignal

from env.off_ppo.SaltEnvUtil import copyScenarioFiles
from env.off_ppo.SaltEnvUtil import getSaRelatedInfo
from env.off_ppo.SaltEnvUtil import getSimulationStartStepAndEndStep
from env.off_ppo.SaltEnvUtil import makePosssibleSaNameList

from env.off_ppo.SappoEnv import SaltSappoEnvV3

from env.off_ppo.SappoRewardMgmt import SaltRewardMgmtV3

from policy.off_ppoTF2 import PPOAgentTF2 #from policy.ppoTF2 import PPOAgentTF2
from ResultCompare import compareResult

from TSOConstants_off_ppo import _FN_PREFIX_, _RESULT_COMP_, _RESULT_COMPARE_SKIP_
from TSOUtil_off_ppo import addArgumentsToParser
from TSOUtil_off_ppo import appendLine
from TSOUtil_off_ppo import convertSaNameToId
from TSOUtil_off_ppo import findOptimalModelNum
from TSOUtil_off_ppo import makeConfigAndProblemVar
from TSOUtil_off_ppo import writeLine



def parseArgument():
    '''
    argument parsing
    :return:
    '''

    parser = argparse.ArgumentParser()

    parser = addArgumentsToParser(parser)

    args = parser.parse_args()

    #args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.mode}.scenario.json"
    #args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.scenario}.scenario.json"
    #args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.mode}_{args.scenario}.scenario.json"

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



def one_hot(indices, depth, on_value=1.0, off_value=0.0):
    scalar = False
    if not isinstance(indices, (list, tuple, np.ndarray)):
        scalar = True
        indices = [indices]

    length = len(indices)
    one_hot = np.ones((length, depth)) * off_value
    one_hot[np.arange(length), indices] = on_value

    if scalar: one_hot = one_hot[0]

    return one_hot


class StateAugmentation:
    
    def __init__(self, step_size, on_value=1.0, off_value=0.0):

        self.step_size = step_size
        self.on_value = on_value
        self.off_value = off_value
        self.time_step = 0
        
    def reset(self):
        self.time_step = 0

        
    def augment(self, state):
        
        time_encoded = one_hot(self.time_step, depth=self.step_size, on_value=self.on_value, off_value=self.off_value)
        #print('state:', state)
        #print('time_encoded:', time_encoded)
        state = np.concatenate([state, time_encoded], axis=-1) 
        
        self.time_step += 1
        
        return state


class Agent:
    
    def __init__(self, env_name, agent_num, action_sizes, state_sizes, ppo_config, problem_var, target_sas, args):

            
        self._init_holder(agent_num, action_sizes)
        self.env_name = env_name
        self.ppo_config = ppo_config
        self.problem_var = problem_var
        
        self.args = args
        
                            
        self.ppo_agent = []
        for i in range(agent_num):
            agent = PPOAgentTF2(env_name, ppo_config, action_sizes[i], state_sizes[i], target_sas[i].strip().replace(' ', '_'))
            self.ppo_agent.append(agent)
                
        

    def _init_holder(self, agent_num, action_sizes):

        actions, logps = [], []
            
        for i in range(agent_num):
            actions.append(list(0 for _ in range(action_sizes[i])))
            logps.append([0])
    
        self._action_holder = actions
        self._logp_holder = logps
        

        
    def act(self, state, info, sampling=True):
        
        idx_of_act_sa = info['idx_of_act_sa']
        
        for i in idx_of_act_sa:
            
            obs = state[i]
            action, logp, mu, std = self.ppo_agent[i].action(obs, sampling)

#            print('mu', mu)
#            print('std', std)
#            print('action:', action)
#            state_action_value = self.ppo_agent[i].evaluate_state_action(obs, action)
#            print('state_action_value:', state_action_value)
                    
            #print('action[0]:', action[0])
            self._action_holder[i] = action[0]
            self._logp_holder[i] = logp[0]
        
        #return self._action_holder, self._logp_holder
        action_holder = copy.deepcopy(self._action_holder)
        logp_holder = copy.deepcopy(self._logp_holder)
        
        return action_holder, logp_holder
        

    

    def store(self, current_state, action, reward, new_state, done, logp, info):

        idx_of_act_sa = info['idx_of_act_sa']
        for i in idx_of_act_sa:
            
            self.ppo_agent[i].memory.store(current_state[i], 
                          action[i], 
                          reward[i], 
                          new_state[i], 
                          done, 
                          logp[i])


    def train(self):

        for agent in self.ppo_agent:
            agent.replayNew()
            
    def getMemorySize(self):
        memory_size = self.ppo_agent[0].memory.getSize()
        return memory_size
    
    
    def save_agent(self, trial):
        args = self.args
        problem_var = self.problem_var
        fn_prefix = "{}/model/{}/{}-{}-trial_{}".format(args.io_home, args.method, args.method.upper(), problem_var, trial)
        for agent in self.ppo_agent:
            agent.saveModel(fn_prefix)

    
    
    def load_agent(self, trial):
        args = self.args
        problem_var = self.problem_var
        fn_prefix = "{}/model/{}/{}-{}-trial_{}".format(args.io_home, args.method, args.method.upper(), problem_var, trial)
        for agent in self.ppo_agent:
            agent.loadModel(fn_prefix)


    

    
class Env(SaltSappoEnvV3):
    
    def __init__(self, args):
        
        args = copy.deepcopy(args)
        args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.mode}_{args.scenario}.scenario.json"
        start_time, end_time = getSimulationStartStepAndEndStep(args)
        trial_len = end_time - start_time

        args.start_time = start_time
        args.end_time = end_time
        
        super(Env, self).__init__(args)
        self._init_holder()

        self.step_size = []
        self.state_augment = []
        for sa_cycle in self.sa_cycle_list: #SappoEnv.py Line 156
            #print('sa_cycle:', sa_cycle)
            step_size = int(trial_len/(sa_cycle * args.control_cycle)) 
            aug = StateAugmentation(step_size, on_value=1.0, off_value=-1.0)
            
            #print('step_size', step_size)
            self.step_size.append(step_size)
            self.state_augment.append(aug)
            

    def get_agent_configuration(self):
        
        env_name = self.env_name
        args = self.args
        ppo_config, problem_var = makeConfigAndProblemVar(self.args)
        agent_num = self.agent_num
        
        action_sizes = []
        state_sizes = []
        target_sas = []
        for i in range(agent_num):
            target_sa = self.sa_name_list[i]

            is_train_target = self.isTrainTarget(target_sa)
            ppo_config["is_train"] = is_train_target

            state_space = self.sa_obj[target_sa]['state_space']
            action_space = self.sa_obj[target_sa]['action_space']

            ##-- TF 2.x : ppo_continuous_hs,py
            action_size = action_space.shape[0]
            #print('action_size', action_size)
            #state_size = (state_space,)
            state_size = (state_space+self.step_size[i],)
            
            action_sizes.append(action_size)
            state_sizes.append(state_size)
            target_sas.append(target_sa)
            
        return env_name, agent_num, action_sizes, state_sizes, ppo_config, problem_var, target_sas, args


    def _init_holder(self):

        # To store reward history of each episode
        self.ep_reward_list = []
    
        #self.current_state = [] # Actually, it is not used. Keep it to maintain consistency with the previous code.

        #actions, logp_ts = [], []
        agent_num = self.agent_num
    
        discrete_actions = []
        
        for i in range(agent_num):
            target_sa = self.sa_name_list[i]
            action_space = self.sa_obj[target_sa]['action_space']
            action_size = action_space.shape[0]
            #actions.append(list(0 for _ in range(action_size)))
            #logp_ts.append([0])
            
            discrete_actions.append(list(0 for _ in range(action_size)))
                # zero because the offset of the fixed signal is used as it is

#        self._action_holder = actions
#        self._logp_holder = logp_ts
        self._discrete_action_holder = discrete_actions
        #self._episodic_agent_reward_holder = [0] * agent_num
        

    def _reshape_state(self, state):

        state = copy.deepcopy(state)
            
        idx_of_act_sa = self.idx_of_act_sa
        for i in idx_of_act_sa:
            obs = state[i]
            obs = self.state_augment[i].augment(obs)
            obs = obs.reshape(1, -1)  # [1,2,3]  ==> [ [1,2,3] ]
            state[i] = obs

        return state
        

    def reset(self):
        
        for aug in self.state_augment: aug.reset()
            
        state = super(Env, self).reset()
        state = self._reshape_state(state)        
        #info = {'idx_of_act_sa':self.idx_of_act_sa} #
        info = {'idx_of_act_sa':copy.deepcopy(self.idx_of_act_sa)} 
        
        #self.current_state = state
        
        
        self.episodic_reward = 0
        self.episodic_agent_reward = [0] * self.agent_num
                
        return state, info

    
    def step(self, actions):
        
        idx_of_act_sa = copy.deepcopy(self.idx_of_act_sa)
        
        for i in idx_of_act_sa:
            
            sa_name = self.sa_name_list[i]
            #print('actions[i]', actions[i])
            
            discrete_action = np.clip(actions[i], -1.0, +1.0) 
            discrete_action = self.action_mgmt.convertToDiscreteAction(sa_name, discrete_action)
            #print('discrete_action', discrete_action)
            self._discrete_action_holder[i] = discrete_action

        #print('self._discrete_action_holder', self._discrete_action_holder)
        
        state, reward, done, info = super(Env, self).step(self._discrete_action_holder) #After calling step(), self.idx_of_act_sa is updated. 
        state = self._reshape_state(state)
        
        idx_of_act_sa = copy.deepcopy(self.idx_of_act_sa)
        # update observation
        for i in idx_of_act_sa:
            #self.current_state[i] = state[i]
            self.episodic_reward += reward[i]
            self.episodic_agent_reward[i] += reward[i]
    
        
        #info['idx_of_act_sa'] = self.idx_of_act_sa 
        info['idx_of_act_sa'] = idx_of_act_sa
    
        if done:
            self.ep_reward_list.append(self.episodic_reward)
            info['episodic_reward'] = self.episodic_reward
            info['recent_returns'] = self.ep_reward_list[-10:]
            info['ma40_reward'] = np.mean(self.ep_reward_list[-40:])
            
        return state, reward, done, info
    


def isolated(conn, args):
    
    env = Env(args)
    
    while True:
        msg = conn.recv()

        
        if msg[0] == 'get_agent_configuration':
            config = env.get_agent_configuration()
            conn.send(config)
            
        elif msg[0] == 'reset':
            state, info = env.reset()       
            conn.send((state, info))
            
        elif msg[0] == 'step':
            state, reward, done, info = env.step(msg[1])

            transition = (state, reward, done, info)
            conn.send(transition)
            
        elif msg[0] == 'close':
            conn.close()
            env.close()
            del env
            break


       
class IsolatedEnv():
    
    def __init__(self, args, max_run=100):

        self.args = copy.deepcopy(args)
        self._max_run = max_run
        self._run = 0
        self._conn = None
        self._env_process = None
        
        self.ep_reward_list = []

        self._create_env_process()

    
    def _create_env_process(self):

        if self._env_process is not None: self.close()
        
        parent_conn, child_conn = Pipe()
        self._conn = parent_conn
        self._env_process = Process(target=isolated, args=(child_conn, self.args))
        #self._env_process.daemon = True
        self._env_process.start()
        

    def get_agent_configuration(self):
        
        self._conn.send(('get_agent_configuration', ))
        config = self._conn.recv()
        return config

        
    def reset(self):
        self._run += 1
        if self._run > self._max_run:
            self._run = 0
            self._create_env_process()
            
            
        self._conn.send(('reset', ))
        state, info = self._conn.recv()

        return state, info


    def step(self, actions):
        
        self._conn.send(('step', actions))
        state, reward, done, info = self._conn.recv()
        
        if done:
            self.ep_reward_list.append(info['episodic_reward'])
            info['recent_returns'] = self.ep_reward_list[-10:]
            info['ma40_reward'] = np.mean(self.ep_reward_list[-40:])
            
        return state, reward, done, info
    
    def close(self):
        #if self.process.is_alive():
        self._conn.send(('close',))
        self._env_process.join()        
        self._conn.close()
        #self.process.terminate()
        
    
        
def run_train_episode(trial, env, agent):

    current_state, info = env.reset()
    done = False
    while not done:
        
        action, logp = agent.act(current_state, info, sampling=True)
        next_state, reward, done, info = env.step(action)
    
        agent.store(current_state, action, reward, next_state, done, logp, info)
        current_state = next_state
        
    
    
    

def run_test_episode(trial, env, agent):

    start_time = time.time()
    state, info = env.reset()
    done = False
    while not done:
        
        action, logp = agent.act(state, info, sampling=False)
        state, reward, done, info = env.step(action)

    end_time = time.time()
    
    print("Reward in current episode:", info['episodic_reward'])
    print('Recent returns:', info['recent_returns'])
    print("Episode * {} * Avg Reward is ==> {}".format(trial, info['ma40_reward']))
    print("Simulation time :", end_time - start_time)
    
    return info

    
def run_valid_episode(trial, env, agent, best_trial, best_score):

    info = run_test_episode(trial, env, agent)    
    score = info['episodic_reward']
    if score > best_score:
        best_trial = trial
        best_score = score
    
    #print('Best trial:', best_trial, best_score)
    
    return best_trial, best_score


def run_multi_thread(trial, envs, agent):

    start_time = time.time()
    num_envs = len(envs)
    threads = []
    for i, env in enumerate(envs):
        thread = Thread(target=run_train_episode, args=(trial*num_envs+i, env, agent))            
        thread.start()
        threads.append(thread)
        
    for thread in threads: thread.join()
    end_time = time.time()

    rewards = [env.ep_reward_list[-1] for env in envs]
    mean = np.mean(rewards)
    std = np.std(rewards)
    
    print("Episode: {}, Simulation time: {}, Memory Size: {}".format(trial, end_time - start_time, agent.getMemorySize()) )
    for i, env in enumerate(envs):
        print('Env:', i)
        print("Reward in current episode:", env.ep_reward_list[-1])
        print('Recent returns:', env.ep_reward_list[-10:])
        print("Avg Reward is ==> {}".format(np.mean(env.ep_reward_list[-40:])))

    return rewards, mean, std

    
def trainSappo(args):
    '''
    model train
      - this is work well with multiple SA
      - infer-TL is considered
    :param args:
    :return:
    '''
    
    valid_args = copy.deepcopy(args)
    valid_args.scenario = '12th'
    
    num_envs = 10
    envs =  []
    for i in range(num_envs):
        env = IsolatedEnv(args)
        envs.append(env)
    
    #create a validation/test evnironment. 
    valid_env = IsolatedEnv(valid_args)
    best_trial = 0; best_score = -np.inf
    
    print('Train:', args.scenario, 'Valid:', valid_args.scenario)
    
    #agent_config = envs[0].get_agent_configuration()
    agent_config = valid_env.get_agent_configuration()
    agent = Agent(*agent_config)

    # fill the replay memory with random plays
    while agent.getMemorySize() < args.mem_len: 
        run_multi_thread(0, envs, agent)
    
    for trial in range(args.epoch):
        run_multi_thread(trial, envs, agent)
        
        start_time = time.time()
        agent.train()
        end_time = time.time()
        print("Training time :", end_time - start_time) 
        
        ### model save
        if trial % args.model_save_period == 0:
            agent.save_agent(trial)
            best_trial, best_score = run_valid_episode(trial, valid_env, agent, best_trial,  best_score)

        print('Best trial:', best_trial, best_score)
       
    optimal_model_num = 0
    return optimal_model_num



def testSappo(args):
    '''
    test trained model

    :param args:
    :return:
    '''
    
    env = Env(args)
    agent_config = env.get_agent_configuration()
    agent = Agent(*agent_config)
    
    if args.action != 'fx': agent.load_agent(trial=args.model_num)
    for trial in range(1):
        run_test_episode(trial, env, agent)

    #env_name, agent_num, action_sizes, state_sizes, ppo_config, problem_var, target_sas, args
    problem_var = agent_config[5]
    # compare traffic simulation results
    if args.result_comp:
        #ft_output = pd.read_csv("{}/output/simulate/{}".format(args.io_home, _RESULT_COMP_.SIMULATION_OUTPUT))
        #rl_output = pd.read_csv("{}/output/test/{}".format(args.io_home, _RESULT_COMP_.SIMULATION_OUTPUT))
        
        ft_output = pd.read_csv("{}/output/simulate/{}/{}".format(args.io_home, args.scenario, _RESULT_COMP_.SIMULATION_OUTPUT))
        rl_output = pd.read_csv("{}/output/test/{}/{}".format(args.io_home, args.scenario, _RESULT_COMP_.SIMULATION_OUTPUT))
        
        comp_skip = _RESULT_COMPARE_SKIP_
        result_fn = compareResultAndStore(args, env, ft_output, rl_output, problem_var, comp_skip)
        __printImprovementRate(env, result_fn, f'Skip {comp_skip} second')

        if DBG_OPTIONS.ResultCompareSkipWarmUp: # comparison excluding warm-up time
            comp_skip = args.warmup_time
            result_fn = compareResultAndStore(args, env, ft_output, rl_output, problem_var, comp_skip)
            __printImprovementRate(env, result_fn, f'Skip {comp_skip} second')

    avg_reward = 0
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
    #output_ft_dir = f'{args.io_home}/output/{args.mode}'
    output_ft_dir = f'{args.io_home}/output/{args.mode}/{args.scenario}'
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
    print(f'launched at {launched}')

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
        print('Mode: train')
        if args.method == 'sappo':
            trainSappo(args)
        else:
            print("internal error : {} is not supported".format(args.method))

    elif args.mode == 'test':
        print('Mode: test')
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