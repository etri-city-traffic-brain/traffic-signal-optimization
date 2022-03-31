import argparse

import numpy as np
import time


from policy.ddqn import DDQN
# from policy.ddpg import DDPG
from policy.ppo import PPOAgent
from policy.ppo_rnd import PPORNDAgent, RunningStats
from test import ddqn_test, sappo_test, ft_simulate, ppornd_test

import keras.backend as K

from config import TRAIN_CONFIG

from env.salt_PennStateAction import SALT_doan_multi_PSA, getScenarioRelatedBeginEndTime
from env.sappo_noConst import SALT_SAPPO_noConst, getScenarioRelatedBeginEndTime
from env.sappo_offset import SALT_SAPPO_offset, getScenarioRelatedBeginEndTime
from env.sappo_offset_single import SALT_SAPPO_offset_single, getScenarioRelatedBeginEndTime
from env.sappo_offset_ea import SALT_SAPPO_offset_EA, getScenarioRelatedBeginEndTime
from env.sappo_green_single import SALT_SAPPO_green_single, getScenarioRelatedBeginEndTime
from env.sappo_green_offset_single import SALT_SAPPO_green_offset_single, getScenarioRelatedBeginEndTime

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test', 'simulate'], default='simulate',
                    help='train - RL model training, test - trained model testing, simulate - fixed-time simulation before test')
parser.add_argument('--model-num', type=str, default='0',
                    help='trained model number for test mode')

parser.add_argument('--method', choices=['sappo', 'ddqn', 'ppornd', 'ppoea'], default='sappo',
                    help='')

parser.add_argument('--map', choices=['dj_all', 'doan', 'sa_1_6_17'], default='sa_1_6_17')

parser.add_argument('--target-TL', type=str, default="SA 1,SA 6,SA 17",
                    help="concatenate signal group with comma(ex. --target-TL SA 101,SA 104)")
# parser.add_argument('--target-TL', type=str, default="SA 6",
#                     help="concatenate signal group with comma(ex. --targetTL SA 101,SA 104)")
parser.add_argument('--start-time', type=int, default=25400)
parser.add_argument('--end-time', type=int, default=32400)

parser.add_argument('--result-comp', type=bool, default=True)

parser.add_argument('--action', choices=['kc', 'offset', 'gr', 'gro'], default='offset',
                    help='kc - keep or change(limit phase sequence), offset - offset, gr - green ratio, gro - green ratio+offset')
parser.add_argument('--state', choices=['v', 'd', 'vd', 'vdd'], default='vdd',
                    help='v - volume, d - density, vd - volume + density, vdd - volume / density')
parser.add_argument('--reward-func', choices=['pn', 'wt', 'wt_max', 'wq', 'wq_median', 'wq_min', 'wq_max', 'wt_SBV', 'wt_SBV_max', 'wt_ABV', 'tt', 'cwq'], default='cwq',
                    help='pn - passed num, wt - wating time, wq - waiting q length, tt - travel time, cwq - cumulative waiting q length')

# dockerize
parser.add_argument('--io-home', type=str, default='.')
parser.add_argument('--scenario-file-path', type=str, default='data/envs/salt/')

### for train
parser.add_argument('--epoch', type=int, default=3000)
parser.add_argument('--warmupTime', type=int, default=600)
parser.add_argument('--model-save-period', type=int, default=20)
parser.add_argument('--printOut', type=bool, default=True, help='print result each step')

### common args
parser.add_argument('--gamma', type=float, default=0.99)

### DDQN args
parser.add_argument('--replay-size', type=int, default=2000)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr-update-period', type=int, default=5)
parser.add_argument('--lr-update-decay', type=float, default=0.9)

### PSA env args
parser.add_argument('--action-t', type=int, default=12)

### PPO args
parser.add_argument('--ppoEpoch', type=int, default=10)
parser.add_argument('--ppo_eps', type=float, default=0.1)
parser.add_argument('--logstdI', type=float, default=0.5)
parser.add_argument('--_lambda', type=float, default=0.95)
parser.add_argument('--a-lr', type=float, default=0.005)
parser.add_argument('--c-lr', type=float, default=0.05)
parser.add_argument('--cp', type=float, default=0.0, help='action change penalty')
parser.add_argument('--mmp', type=float, default=1.0, help='min max penalty')
parser.add_argument('--actionp', type=float, default=0.2, help='action 0 or 1 prob.(-1~1): Higher value_collection select more zeros')

### PPO RND
parser.add_argument('--gamma-i', type=float, default=0.11)

### PPO + RESNET
parser.add_argument('--res', type=bool, default=True)

### PPO + Memory
parser.add_argument('--memLen', type=int, default=1000, help='memory length')
parser.add_argument('--memFR', type=float, default=0.9, help='memory forget ratio')

### SAPPO OFFSET
parser.add_argument('--offsetrange', type=int, default=2, help="offset side range")
parser.add_argument('--controlcycle', type=int, default=5)

### GREEN RATIO args
parser.add_argument('--addTime', type=int, default=2)

args = parser.parse_args()

args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.mode}.scenario.json"

if args.map == 'sa_1_6_17' or args.map=='dj_all':
    args.trainStartTime = 25200 # 07:00
    args.trainEndTime = 32400   # 09:00
    args.testStartTime = 25200  # 07:00
    args.testEndTime = 32400    # 09:00

problem_var = ""
# problem_var = "tau{}".format(args.tau)
problem_var = "_gamma{}".format(args.gamma)
# problem_var += "_yp0_actionT_{}".format(args.action_t)
# problem_var += "_map_{}".format(args.map)
# problem_var += "_method_{}".format(args.method)
problem_var += "_resnet_{}".format(args.res)
# problem_var += "_state_{}".format(args.state)
# problem_var += "_reward_{}".format(args.reward_func)
problem_var += "_action_{}".format(args.action)
problem_var += "_netsize_{}".format(TRAIN_CONFIG['network_size'])
# problem_var += "_gamma_{}".format(args.gamma)
# problem_var += "_lambda_{}".format(args._lambda)
problem_var += "_ppoEpoch_{}".format(args.ppoEpoch)
problem_var += "_ppoeps_{}".format(args.ppo_eps)
# problem_var += "_lr_{}".format(args.lr)
problem_var += "_alr_{}".format(args.a_lr)
problem_var += "_clr_{}".format(args.c_lr)
# problem_var += "_cc_{}".format(args.controlcycle)
problem_var += "_mLen_{}".format(args.memLen)
problem_var += "_mFR_{}".format(args.memFR)
# problem_var += "_offsetrange_{}".format(args.offsetrange)
# problem_var += "_logstdI_{}".format(args.logstdI)

if args.method=='ppornd':
    problem_var += "_gammai_{}".format(args.gamma_i)
    problem_var += "_rndnetsize_{}".format(TRAIN_CONFIG['rnd_network_size'])
if args.method=='ppoea':
    problem_var += "_ppoEpoch_{}".format(args.ppoEpoch)
    problem_var += "_ppoeps_{}".format(args.ppo_eps)
if len(args.target_TL.split(","))==1:
    problem_var += "_{}".format(args.target_TL.split(",")[0])

if args.action == 'gr' or args.action == 'gro':
    problem_var += "_addTime_{}".format(args.addTime)

io_home = args.io_home
output_train_dir = '{}/output/train'.format(io_home)
fn_train_epoch_total_reward = "{}/train_epoch_total_reward.txt".format(output_train_dir)
fn_train_epoch_tl_reward = "{}/train_epoch_tl_reward.txt".format(output_train_dir)

def makeDirectories(dir_name_list):
    import os
    for dir_name in dir_name_list:
        os.makedirs(dir_name, exist_ok=True)
    return

def run_ddqn():
    import tensorflow as tf
    problem = "DDQN_SALT_doan_discrete_PSA_" + problem_var
    env = SALT_doan_multi_PSA(args)

    trials = args.epoch

    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    lr_update_period = args.lr_update_period
    lr_update_decay = args.lr_update_decay

    state_weight = env.state_weight

    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    train_summary_writer = tf.summary.create_file_writer('{}/logs/DDQN/{}/{}'.format(io_home, problem, time_data))

    f = open(fn_train_epoch_total_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    f.write('epoch,reward,40ep_reward\n')
    f.close()

    f = open(fn_train_epoch_tl_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    f.write('epoch,tl_name,reward,40ep_reward\n')
    f.close()

    agent_num = env.agent_num

    # updateTargetNetwork = 1000
    dqn_agent = []
    ep_agent_reward_list = []
    agent_crossName = []
    for i in range(agent_num):
        target_tl = list(env.target_tl_obj.keys())[i]
        state_space = env.target_tl_obj[target_tl]['state_space']
        action_space = env.target_tl_obj[target_tl]['action_space']
        dqn_agent.append(DDQN(args = args, env=env, state_space=state_space, action_space=action_space))
        ep_agent_reward_list.append([])
        agent_crossName.append(env.target_tl_obj[target_tl]['crossName'])

    print(dqn_agent)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    steps = []

    for trial in range(trials):

        with train_summary_writer.as_default():
            tf.summary.scalar('train/epsilon', dqn_agent[0].epsilon, step=trial)

        actions = [0] * agent_num
        cur_state = env.reset()
        episodic_reward = 0
        episodic_agent_reward = [0]*agent_num
        start = time.time()
        for step in range(trial_len):
            for i in range(agent_num):
                actions[i] = dqn_agent[i].act(cur_state[i])

            new_state, reward, done, _ = env.step(actions)
            for i in range(agent_num):
                new_state[i] = new_state[i]
                dqn_agent[i].remember(cur_state[i], actions[i], reward[i], new_state[i], done)

                dqn_agent[i].replay()  # internally iterates default (prediction) model
                dqn_agent[i].target_train()  # iterates target model

                cur_state[i] = new_state[i]
                episodic_reward += reward[i]
                episodic_agent_reward[i] += reward[i]

            if done:
                break

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        ma1_reward = np.mean(ep_reward_list[-1:])
        print("Episode * {} * Avg Reward is ==> {}".format(trial, avg_reward))
        print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        avg_reward_list.append(avg_reward)

        with train_summary_writer.as_default():
            tf.summary.scalar('train/reward_40ep_mean', avg_reward, step=trial)
            tf.summary.scalar('train/reward', ma1_reward, step=trial)
            tf.summary.scalar('train/lr', dqn_agent[0].model.optimizer.lr.numpy(), step=trial)
            for i in range(agent_num):
                ep_agent_reward_list[i].append(episodic_agent_reward[i])
                tf.summary.scalar('train_agent_reward_40ep_mean/agent_{}_{}'.format(i,agent_crossName[i]), np.mean(ep_agent_reward_list[i][-40:]), step=trial)
                tf.summary.scalar('train_agent_reward/agent_{}_{}'.format(i,agent_crossName[i]), np.mean(ep_agent_reward_list[i][-1:]), step=trial)
            # tf.summary.scalar('train/epsilon', dqn_agent[0].epsilon, step=trial)

        f = open(fn_train_epoch_total_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)
        f.write('{},{},{}\n'.format(trial, ma1_reward, avg_reward))
        f.close()

        for i in range(agent_num):
            f = open(fn_train_epoch_tl_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                     newline=None,
                     closefd=True, opener=None)
            f.write('{},{},{},{}\n'.format(trial, agent_crossName[i], np.mean(ep_agent_reward_list[i][-1:]), np.mean(ep_agent_reward_list[i][-40:])))
            f.close()

        if trial > 0 and trial % lr_update_period == 0:
            for i in range(agent_num):
                K.set_value(dqn_agent[i].model.optimizer.lr, dqn_agent[i].model.optimizer.lr * lr_update_decay)
                print("optimizer lr ", round(dqn_agent[i].model.optimizer.lr.numpy(), 5))

        if trial % args.model_save_period == 0:
            for i in range(agent_num):
                dqn_agent[i].save_model("{}/model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(io_home, problem_var, i, trial))

def run_sappo():
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    # load envs
    # args.problem - tensorboard 기록용
    if args.action=='kc':
        print("SAPPO KEEP OR CHANGE")
        args.problem = "SAPPO_NoConstraints_" + problem_var
        env = SALT_SAPPO_noConst(args)
    if args.action=='offset':
        if len(args.target_TL.split(",")) == 1: # target-TL에 교차로 그룹 하나 들어올 때(ex. SA 6)
            print("SAPPO OFFSET SINGLE")
            args.problem = "SAPPO_offset_single_" + problem_var
            env = SALT_SAPPO_offset_single(args)
        else:                                   # target-TL에 여러 개 교차로 그룹이 들어올 때(ex. SA 1,SA 6,SA 17)
            print("SAPPO OFFSET")
            args.problem = "SAPPO_offset_" + problem_var
            env = SALT_SAPPO_offset(args)
    if args.action=='gr':
        print("SAPPO GREEN RATIO")
        args.problem = "SAPPO_GR_single" + problem_var
        env = SALT_SAPPO_green_single(args)
    if args.action=='gro':
        print("SAPPO GREEN RATIO + OFFSET")
        args.problem = "SAPPO_GRO_single" + problem_var
        env = SALT_SAPPO_green_offset_single(args)

    trials = args.epoch

    ### get simulation start & end time from scenario file
    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    ### 훈련 시작 시간 기록
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    train_summary_writer = tf.summary.FileWriter('{}/logs/SAPPO/{}/{}'.format(io_home, args.problem, time_data))

    ### 가시화 서버에서 사용할 epoch별 전체 보상 파일
    f = open(fn_train_epoch_total_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    f.write('epoch,reward,40ep_reward\n')
    f.close()
    ### 가시화 서버에서 사용할 epoch별 agent별 전체 보상 파일
    f = open(fn_train_epoch_tl_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    f.write('epoch,tl_name,reward,40ep_reward\n')
    f.close()

    ep_agent_reward_list = []
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    ma40_reward_list = []

    agent_crossName = []
    agent_reward1, agent_reward40 = [], []
    agent_reward1_summary, agent_reward40_summary = [], []

    total_reward = tf.Variable(0, dtype=tf.float32)
    total_reward_summary = tf.summary.scalar('train/reward', total_reward)

    ppo_agent = []
    agent_num = env.agent_num
    for i in range(agent_num):
        target_sa = list(env.sa_obj.keys())[i]
        state_space = env.sa_obj[target_sa]['state_space']
        action_space = env.sa_obj[target_sa]['action_space']
        action_min = env.sa_obj[target_sa]['action_min']
        action_max = env.sa_obj[target_sa]['action_max']
        print(f"{target_sa}, state space {state_space} action space {action_space}, action min {action_min}, action max {action_max}")
        ppo_agent.append(PPOAgent(args=args, state_space=state_space, action_space=action_space, action_min=action_min, action_max=action_max, agentID=i))
        ep_agent_reward_list.append([])
        agent_crossName.append(env.sa_obj[target_sa]['crossName_list'])

        agent_reward1.append(tf.Variable(0, dtype=tf.float32))
        agent_reward40.append(tf.Variable(0, dtype=tf.float32))
        agent_reward1_summary.append(tf.summary.scalar('train_agent_reward/agent_{}'.format(list(env.sa_obj.keys())[i]), agent_reward1[i]))  # summary to write to TensorBoard
        agent_reward40_summary.append(tf.summary.scalar('train_agent_reward_40ep_mean/agent_{}'.format(list(env.sa_obj.keys())[i]), agent_reward40[i]))  # summary to write to TensorBoard

    print(ppo_agent)

    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ### agent 수 만큼 collection 생성
    actions_collection, state_collection, value_collection, logp_t_collection, done_collection, reward_collection = [], [], [], [], [], []
    for target_sa in env.sa_obj:
        actions_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0]) # 에이전트의 action이 여러 개이므로 에이전트의 action 수 만큼 0으로 초기화
        ### 나머지 값들은 실수 값이라 0으로만 초기화
        state_collection.append([0])
        value_collection.append([0])
        logp_t_collection.append([0])
        done_collection.append([0])
        reward_collection.append([0])

    for trial in range(trials):
        actions, v_t, logp_t = [], [], []
        next_values, adv, target = [], [], []

        cur_state = env.reset()

        sa_cycle = []

        ### agent 수 만큼 필요한 변수들 초기화
        for target_sa in env.sa_obj:
            actions.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            v_t.append([0])
            logp_t.append([0])

            next_values.append([0])
            adv.append([0])
            target.append([0])

            sa_cycle = np.append(sa_cycle, env.sa_obj[target_sa]['cycle_list'][0])

        episodic_reward = 0
        episodic_agent_reward = [0]*agent_num
        start = time.time()
        m_len = args.memLen
        m_remove_ratio = args.memFR
        print("trial_len", trial_len)
        for t in range(trial_len):
            discrete_actions = []
            for i in range(agent_num):
                actions[i], v_t[i], logp_t[i] = ppo_agent[i].get_action([cur_state[i]], sess)
                actions[i], v_t[i], logp_t[i] = actions[i][0], v_t[i][0], logp_t[i][0]

                target_sa = list(env.sa_obj.keys())[i]
                discrete_action = []
                for di in range(len(actions[i])):
                    # discrete_action.append(np.digitize(actions[i][di], bins=env.sa_obj[target_sa]['duration_bins_list'][di]))
                    if args.action=='kc':
                        discrete_action.append(0 if actions[i][di] < args.actionp else 1)
                    if args.action=='offset':
                        # discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2))
                        discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2/args.offsetrange))
                    if args.action=='gr':
                        discrete_action.append(np.digitize(actions[i][di], bins=np.linspace(-1, 1, len(env.sa_obj[target_sa]['action_list_list'][di]))) - 1)

                if args.action == 'gro':
                    for di in range(int(len(actions[i])/2)):
                        discrete_action.append(int(np.round(actions[i][di * 2] * sa_cycle[i]) / 2 / args.offsetrange))
                        discrete_action.append(np.digitize(actions[i][di * 2 + 1], bins=np.linspace(-1, 1, len(env.sa_obj[target_sa]['action_list_list'][di]))) - 1)

                discrete_actions.append(discrete_action)
            # print("discrete_actions", discrete_actions)

            new_state, reward, done, _ = env.step(discrete_actions)

            if len(args.target_TL.split(",")) == 1:
                print(f"t{t} current state mean {np.mean(cur_state)} action {np.round(actions, 2)} reward {reward} new_state_mean {np.mean(new_state)}")

                for i in range(agent_num):
                    if trial==0:
                        state_collection[i] = np.r_[state_collection[i], [cur_state[i]]] if t else [cur_state[i]]
                        actions_collection[i] = np.r_[actions_collection[i], [actions[i]]] if t else [actions[i]]
                        value_collection[i] = np.r_[value_collection[i], v_t[i]] if t else [v_t[i]]
                        logp_t_collection[i] = np.r_[logp_t_collection[i], logp_t[i]] if t else [logp_t[i]]
                        done_collection[i] = np.r_[done_collection[i], done] if t else [done]
                        reward_collection[i] = np.r_[reward_collection[i], reward[i]] if t else [reward[i]]
                    else:
                        if len(state_collection[i]) >= m_len:
                            nrc = np.random.choice(range(m_len), int(m_len*m_remove_ratio), replace=False)
                            state_collection[i] = np.delete(state_collection[i], nrc, axis=0)
                            actions_collection[i] = np.delete(actions_collection[i], nrc, axis=0)
                            value_collection[i] = np.delete(value_collection[i], nrc, axis=0)
                            logp_t_collection[i] = np.delete(logp_t_collection[i], nrc, axis=0)
                            done_collection[i] = np.delete(done_collection[i], nrc, axis=0)
                            reward_collection[i] = np.delete(reward_collection[i], nrc, axis=0)

                        state_collection[i] = np.r_[state_collection[i], [cur_state[i]]]
                        actions_collection[i] = np.r_[actions_collection[i], [actions[i]]]
                        value_collection[i] = np.r_[value_collection[i], v_t[i]]
                        logp_t_collection[i] = np.r_[logp_t_collection[i], logp_t[i]]
                        done_collection[i] = np.r_[done_collection[i], done]
                        reward_collection[i] = np.r_[reward_collection[i], reward[i]]

                    # Update the observation
                    cur_state[i] = new_state[i]

                    episodic_reward += reward[i]
                    episodic_agent_reward[i] += reward[i]
            else:
                if t % int(sa_cycle[i] * args.controlcycle) == 0:
                    for i in range(agent_num):
                        if trial==0:
                            state_collection[i] = np.r_[state_collection[i], [cur_state[i]]] if t else [cur_state[i]]
                            actions_collection[i] = np.r_[actions_collection[i], [actions[i]]] if t else [actions[i]]
                            value_collection[i] = np.r_[value_collection[i], v_t[i]] if t else [v_t[i]]
                            logp_t_collection[i] = np.r_[logp_t_collection[i], logp_t[i]] if t else [logp_t[i]]
                            done_collection[i] = np.r_[done_collection[i], done] if t else [done]
                            reward_collection[i] = np.r_[reward_collection[i], reward[i]] if t else [reward[i]]
                        else:
                            if len(state_collection[i]) >= m_len:
                                nrc = np.random.choice(range(m_len), int(m_len*m_remove_ratio), replace=False)
                                state_collection[i] = np.delete(state_collection[i], nrc, axis=0)
                                actions_collection[i] = np.delete(actions_collection[i], nrc, axis=0)
                                value_collection[i] = np.delete(value_collection[i], nrc, axis=0)
                                logp_t_collection[i] = np.delete(logp_t_collection[i], nrc, axis=0)
                                done_collection[i] = np.delete(done_collection[i], nrc, axis=0)
                                reward_collection[i] = np.delete(reward_collection[i], nrc, axis=0)

                            state_collection[i] = np.r_[state_collection[i], [cur_state[i]]]
                            actions_collection[i] = np.r_[actions_collection[i], [actions[i]]]
                            value_collection[i] = np.r_[value_collection[i], v_t[i]]
                            logp_t_collection[i] = np.r_[logp_t_collection[i], logp_t[i]]
                            done_collection[i] = np.r_[done_collection[i], done]
                            reward_collection[i] = np.r_[reward_collection[i], reward[i]]

                        # Update the observation
                        cur_state[i] = new_state[i]

                        episodic_reward += reward[i]
                        episodic_agent_reward[i] += reward[i]
            if done:
                break

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        ma1_reward = np.mean(ep_reward_list[-1:])
        ma40_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {} MemoryLen {}".format(trial, ma40_reward, len(state_collection[0])))
        print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

        ### 전체 평균 보상 tensorboard에 추가
        ma40_reward_list.append(ma40_reward)
        sess.run(total_reward.assign(ma1_reward))
        train_summary_writer.add_summary(sess.run(total_reward_summary), trial)

        for i in range(agent_num):
            print("update")
            ### ppo agent update
            v_t[i] = ppo_agent[i].get_action([cur_state[i]], sess)[1][0]
            value_collection[i] = np.r_[value_collection[i], v_t[i]]
            next_values[i] = np.copy(value_collection[i][1:])
            value_collection[i] = value_collection[i][:-1]
            adv[i], target[i] = ppo_agent[i].get_gaes(reward_collection[i], done_collection[i], value_collection[i], next_values[i], True)
            ppo_agent[i].update(state_collection[i], actions_collection[i], target[i], adv[i], logp_t_collection[i], sess)

            ### epoch별, 에이전트 별 평균 보상 & 40epoch 평균 보상 tensorboard에 추가
            ep_agent_reward_list[i].append(episodic_agent_reward[i]) # epoisode별 리워드 리스트에 저장
            sess.run(agent_reward1[i].assign(np.mean(ep_agent_reward_list[i][-1:])))
            sess.run(agent_reward40[i].assign(np.mean(ep_agent_reward_list[i][-40:])))
            train_summary_writer.add_summary(sess.run(agent_reward1_summary[i]), trial)
            train_summary_writer.add_summary(sess.run(agent_reward40_summary[i]), trial)

        train_summary_writer.flush() # update tensorboard

        ### 가시화 서버에서 사용할 epoch별 reward 파일
        f = open(fn_train_epoch_total_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)
        f.write('{},{},{}\n'.format(trial, ma1_reward, ma40_reward))
        f.close()

        ### 가시화 서버에서 사용할 epoch별 agent별 reward 파일
        for i in range(agent_num):
            f = open(fn_train_epoch_tl_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                     newline=None,
                     closefd=True, opener=None)
            f.write('{},{},{},{}\n'.format(trial, agent_crossName[i], np.mean(ep_agent_reward_list[i][-1:]), np.mean(ep_agent_reward_list[i][-40:])))
            f.close()

        ### model save
        if trial % args.model_save_period == 0:
            fn = "{}/model/sappo/SAPPO-{}-trial".format(io_home, problem_var)
            saver.save(sess, fn, global_step=trial)

def run_ppornd():
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    def state_next_normalize(sample_size, running_stats_s_):
        buffer_s_ = []
        s = env.reset()
        for i in range(sample_size):
            a = env.action_space.sample()
            s_, r, done, _ = env.step(a)
            buffer_s_.append(s_)
        running_stats_s_.update(np.array(buffer_s_))

    # normalize & clip a running buffer
    # used on extrinsic reward (buffer_r), intrinsic reward (buffer_r_i) & next state(s_)
    def running_stats_fun(run_stats, buf, clip, clip_state):
        run_stats.update(np.array(buf))
        buf = (np.array(buf) - run_stats.mean) / run_stats.std
        if clip_state == True:
            buf = np.clip(buf, -clip, clip)
        return buf

    if args.action=='kc':
        args.problem = "PPORND_NoConstraints_" + problem_var
        env = SALT_SAPPO_noConst(args)
    if args.action=='offset':
        args.problem = "PPORND_offset_" + problem_var
        env = SALT_SAPPO_offset(args)

    if len(args.target_TL.split(","))==1:
        args.problem = "PPORND_offset_single_" + problem_var
        env = SALT_SAPPO_offset_single(args)

    trials = args.epoch
    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    # self.train_summary_writer = tf.summary.FileWriter('logs/SAPPO/{}/{}_{}'.format(args.problem, self.time_data, agentID))
    # train_summary_writer = tf.summary.FileWriter('logs/SAPPO/{}/{}'.format(args.problem, time_data))

    train_summary_writer = tf.summary.FileWriter('{}/logs/PPORND/{}/{}'.format(io_home, args.problem, time_data))

    f = open(fn_train_epoch_total_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    f.write('epoch,reward,40ep_reward\n')
    f.close()

    f = open(fn_train_epoch_tl_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    f.write('epoch,tl_name,reward,40ep_reward\n')
    f.close()

    agent_num = env.agent_num

    ppornd_agent = []
    ep_agent_reward_list = []
    agent_crossName = []
    agent_reward1 = []
    agent_reward40 = []
    agent_reward1_summary = []
    agent_reward40_summary = []

    total_reward = tf.Variable(0, dtype=tf.float32)
    total_reward_summary = tf.summary.scalar('train/reward', total_reward)

    running_stats_s = RunningStats()
    running_stats_s_ = RunningStats()
    running_stats_r = RunningStats()
    running_stats_r_i = RunningStats()

    for i in range(agent_num):
        target_sa = list(env.sa_obj.keys())[i]
        state_space = env.sa_obj[target_sa]['state_space']
        action_space = env.sa_obj[target_sa]['action_space']
        action_min = env.sa_obj[target_sa]['action_min']
        action_max = env.sa_obj[target_sa]['action_max']
        print(f"{target_sa}, state space {state_space} action space {action_space}, action min {action_min}, action max {action_max}")
        ppornd_agent.append(PPORNDAgent(args=args, state_space=state_space, action_space=action_space, action_min=action_min, action_max=action_max, agentID=i))
        ep_agent_reward_list.append([])
        agent_crossName.append(env.sa_obj[target_sa]['crossName_list'])

        agent_reward1.append(tf.Variable(0, dtype=tf.float32))
        agent_reward40.append(tf.Variable(0, dtype=tf.float32))
        agent_reward1_summary.append(tf.summary.scalar('train_agent_reward/agent_{}'.format(list(env.sa_obj.keys())[i]), agent_reward1[i]))  # summary to write to TensorBoard
        agent_reward40_summary.append(tf.summary.scalar('train_agent_reward_40ep_mean/agent_{}'.format(list(env.sa_obj.keys())[i]), agent_reward40[i]))  # summary to write to TensorBoard

    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(ppornd_agent)

    print("env.sa_obj", env.sa_obj)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    ma40_reward_list = []

    for trial in range(trials):
        buffer_Vs, buffer_V_is = [], []

        actions = []

        actions_collection = []
        state_collection = []
        next_state_collection = []
        value_collection = []
        logp_t_collection = []
        done_collection = []
        reward_collection = []

        v_t = []
        next_values = []
        adv = []
        target = []

        sa_cycle = []

        for target_sa in env.sa_obj:
            actions.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            buffer_Vs.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            buffer_V_is.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])

            actions_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            state_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            next_state_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            value_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            logp_t_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            done_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            reward_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])

            v_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            next_values.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            adv.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            target.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            sa_cycle = np.append(sa_cycle, env.sa_obj[target_sa]['cycle_list'][0])

        cur_state = env.reset()
        episodic_reward = 0
        episodic_agent_reward = [0]*agent_num
        start = time.time()
        for t in range(trial_len):
            discrete_actions = []
            for i in range(agent_num):
                actions[i] = ppornd_agent[i].choose_action([cur_state[i]], sess)

                discrete_action = []
                for di in range(len(actions[i])):
                    if args.action=='kc':
                        discrete_action.append(0 if actions[i][di] < args.actionp else 1)
                    if args.action=='offset':
                        # discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2))
                        discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2/args.offsetrange))

                discrete_actions.append(discrete_action)

            new_state, reward, done, _ = env.step(discrete_actions)

            if len(args.target_TL.split(",")) == 1:
                for i in range(agent_num):
                    state_collection[i] = np.r_[state_collection[i], [cur_state[i]]] if t else [cur_state[i]]
                    actions_collection[i] = np.r_[actions_collection[i], [actions[i]]] if t else [actions[i]]
                    done_collection[i] = np.r_[done_collection[i], done] if t else [done]
                    reward_collection[i] = np.r_[reward_collection[i], reward[i] * ppornd_agent[i].ext_r_coeff] if t else [reward[i]]
                    v = ppornd_agent[i].get_v([cur_state[i]], sess)
                    buffer_Vs[i] = np.r_[buffer_Vs[i], v] if t else [v]
                    v_i = ppornd_agent[i].get_v_i([cur_state[i]], sess)
                    buffer_V_is[i] = np.r_[buffer_V_is[i], v_i] if t else [v_i]
                    next_state_collection[i] = np.r_[next_state_collection[i], [new_state[i]]] if t else [new_state[i]]

                    cur_state[i] = new_state[i]

                    episodic_reward += reward[i]
                    episodic_agent_reward[i] += reward[i]
            else:
                if t % int(sa_cycle[i] * args.controlcycle) == 0:
                    for i in range(agent_num):
                        state_collection[i] = np.r_[state_collection[i], [cur_state[i]]] if t else [cur_state[i]]
                        actions_collection[i] = np.r_[actions_collection[i], [actions[i]]] if t else [actions[i]]
                        done_collection[i] = np.r_[done_collection[i], done] if t else [done]
                        reward_collection[i] = np.r_[reward_collection[i], reward[i] * ppornd_agent[i].ext_r_coeff] if t else [reward[i]]
                        v = ppornd_agent[i].get_v([cur_state[i]], sess)
                        buffer_Vs[i] = np.r_[buffer_Vs[i], v] if t else [v]
                        v_i = ppornd_agent[i].get_v_i([cur_state[i]], sess)
                        buffer_V_is[i] = np.r_[buffer_V_is[i], v_i] if t else [v_i]

                        # Update the observation
                        cur_state[i] = new_state[i]

                        episodic_reward += reward[i]
                        episodic_agent_reward[i] += reward[i]

            if done:
                break
        print("buffer_Vs", buffer_Vs)
        print("buffer_V_is", buffer_V_is)
        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        ma40_reward = np.mean(ep_reward_list[-40:])
        ma1_reward = np.mean(ep_reward_list[-1:])
        print("Episode * {} * Avg Reward is ==> {}".format(trial, ma40_reward))
        print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        ma40_reward_list.append(ma40_reward)

        sess.run(total_reward.assign(ma1_reward))  # update accuracy variable
        train_summary_writer.add_summary(sess.run(total_reward_summary), trial)  # add summary

        for i in range(agent_num):
            buffer_r_i = ppornd_agent[i].intrinsic_r(next_state_collection[i], sess)
            # Batch normalize running extrinsic r
            reward_collection[i] = running_stats_fun(running_stats_r, reward_collection[i], ppornd_agent[i].r_CLIP, False)
            # Batch normalize running intrinsic r_i
            buffer_r_i = running_stats_fun(running_stats_r_i, buffer_r_i, ppornd_agent[i].r_CLIP, False)

            v_s_ = ppornd_agent[i].get_v([cur_state[i]], sess)
            tdlamret, adv = ppornd_agent[i].add_vtarg_and_adv(np.vstack(reward_collection[i]),
                                                      np.vstack(done_collection[i]),
                                                      np.vstack(buffer_Vs[i]),
                                                      v_s_,
                                                      args.gamma,
                                                      ppornd_agent[i].lamda)

            v_s_i = ppornd_agent[i].get_v_i([cur_state[i]], sess)
            tdlamret_i, adv_i = ppornd_agent[i].add_vtarg_and_adv(np.vstack(buffer_r_i),
                                                          np.vstack(done_collection[i]),
                                                          np.vstack(buffer_V_is[i]),
                                                          v_s_i,
                                                          args.gamma_i,
                                                          ppornd_agent[i].lamda)

            bs, bs_, ba, br, br_i, b_adv = np.vstack(state_collection[i]), np.vstack(next_state_collection[i]), np.vstack(actions_collection[i]), tdlamret, tdlamret_i, np.vstack(adv + adv_i)  # sum advantages

            ppornd_agent[i].update(bs, bs_, ba, br, br_i, b_adv, sess)

            ep_agent_reward_list[i].append(episodic_agent_reward[i])

            sess.run(agent_reward1[i].assign(np.mean(ep_agent_reward_list[i][-1:])))  # update accuracy variable
            sess.run(agent_reward40[i].assign(np.mean(ep_agent_reward_list[i][-40:])))  # update accuracy variable

            train_summary_writer.add_summary(sess.run(agent_reward1_summary[i]), trial)  # add summary
            train_summary_writer.add_summary(sess.run(agent_reward40_summary[i]), trial)  # add summary

        train_summary_writer.flush()

        f = open(fn_train_epoch_total_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)
        f.write('{},{},{}\n'.format(trial, ma1_reward, ma40_reward))
        f.close()

        for i in range(agent_num):
            f = open(fn_train_epoch_tl_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                     newline=None,
                     closefd=True, opener=None)
            f.write('{},{},{},{}\n'.format(trial, agent_crossName[i], np.mean(ep_agent_reward_list[i][-1:]), np.mean(ep_agent_reward_list[i][-40:])))
            f.close()

        if trial % args.model_save_period == 0:
            fn = "{}/model/ppornd/PPORND-{}-trial".format(io_home, problem_var)
            saver.save(sess, fn, global_step=trial)

def run_ppoea():
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    if args.action=='offset':
        args.problem = "PPOEA_offset_" + problem_var
        env = SALT_SAPPO_offset_EA(args)

    if len(args.target_TL.split(","))==1:
        args.problem = "PPOEA_offset_single_" + problem_var
        env = SALT_SAPPO_offset_EA(args)

    trials = args.epoch
    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))

    train_summary_writer = tf.summary.FileWriter('{}/logs/PPOEA/{}/{}'.format(io_home, args.problem, time_data))

    f = open(fn_train_epoch_total_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    f.write('epoch,reward,40ep_reward\n')
    f.close()

    f = open(fn_train_epoch_tl_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    f.write('epoch,tl_name,reward,40ep_reward\n')
    f.close()

    agent_num = env.agent_num

    ppornd_agent = []
    ep_agent_reward_list = []
    agent_crossName = []
    agent_reward1 = []
    agent_reward40 = []
    agent_reward1_summary = []
    agent_reward40_summary = []

    total_reward = tf.Variable(0, dtype=tf.float32)
    total_reward_summary = tf.summary.scalar('train/reward', total_reward)

    # print("agent_num", agent_num)
    for i in range(agent_num):
        target_sa = list(env.sa_obj.keys())[i]
        state_space = env.sa_obj[target_sa]['state_space']
        action_space = env.sa_obj[target_sa]['action_space']
        action_min = env.sa_obj[target_sa]['action_min']
        action_max = env.sa_obj[target_sa]['action_max']
        print(f"{target_sa}, state space {state_space} action space {action_space}, action min {action_min}, action max {action_max}")
        ppornd_agent.append(PPOAgent(args=args, state_space=state_space, action_space=action_space, action_min=action_min, action_max=action_max, agentID=i))
        ep_agent_reward_list.append([])
        agent_crossName.append(env.sa_obj[target_sa]['crossName_list'])

        agent_reward1.append(tf.Variable(0, dtype=tf.float32))
        agent_reward40.append(tf.Variable(0, dtype=tf.float32))
        agent_reward1_summary.append(tf.summary.scalar('train_agent_reward/agent_{}'.format(list(env.sa_obj.keys())[i]), agent_reward1[i]))  # summary to write to TensorBoard
        agent_reward40_summary.append(tf.summary.scalar('train_agent_reward_40ep_mean/agent_{}'.format(list(env.sa_obj.keys())[i]), agent_reward40[i]))  # summary to write to TensorBoard

    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(ppornd_agent)
    print("env.sa_obj", env.sa_obj)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    ma40_reward_list = []

    for trial in range(trials):
        actions = []
        v_t = []
        logp_t = []

        actions_collection = []
        state_collection = []
        value_collection = []
        logp_t_collection = []
        done_collection = []
        reward_collection = []

        v_t = []
        next_values = []
        adv = []
        target = []

        sa_cycle = []

        for target_sa in env.sa_obj:
            actions.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            v_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            logp_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])

            actions_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            state_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            value_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            logp_t_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            done_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            reward_collection.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])

            v_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            next_values.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            adv.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            target.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            sa_cycle = np.append(sa_cycle, env.sa_obj[target_sa]['cycle_list'][0])

        cur_state = env.reset()
        episodic_reward = 0
        episodic_agent_reward = [0]*agent_num
        start = time.time()
        print("trial_len", trial_len)
        for t in range(trial_len):
            discrete_actions = []
            # print("range(agent_num)", range(agent_num))
            for i in range(agent_num):
                actions[i], v_t[i], logp_t[i] = ppornd_agent[i].get_action([cur_state[i]], sess)
                actions[i], v_t[i], logp_t[i] = actions[i][0], v_t[i][0], logp_t[i][0]

                target_sa = list(env.sa_obj.keys())[i]
                discrete_action = []
                for di in range(len(actions[i])):
                    # discrete_action.append(np.digitize(actions[i][di], bins=env.sa_obj[target_sa]['duration_bins_list'][di]))
                    if args.action=='kc':
                        discrete_action.append(0 if actions[i][di] < args.actionp else 1)
                    if args.action=='offset':
                        # discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2))
                        discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2/args.offsetrange))

                discrete_actions.append(discrete_action)
            new_state, reward, done, _, virtual_actions = env.step(discrete_actions)
            # print(f"RUN ACTION : {actions} VIRTUAL ACTION : {virtual_actions} REWARD : {reward}")

            if len(args.target_TL.split(",")) == 1:
                for i in range(agent_num):
                    state_collection[i] = np.r_[state_collection[i], [cur_state[i]]] if t else [cur_state[i]]
                    actions_collection[i] = np.r_[actions_collection[i], [virtual_actions[i]]] if t else [virtual_actions[i]]
                    value_collection[i] = np.r_[value_collection[i], v_t[i]] if t else [v_t[i]]
                    logp_t_collection[i] = np.r_[logp_t_collection[i], logp_t[i]] if t else [logp_t[i]]
                    done_collection[i] = np.r_[done_collection[i], done] if t else [done]
                    reward_collection[i] = np.r_[reward_collection[i], reward[i]] if t else [reward[i]]

                    # Update the observation
                    cur_state[i] = new_state[i]

                    episodic_reward += reward[i]
                    episodic_agent_reward[i] += reward[i]
            else:
                if t % int(sa_cycle[i] * args.controlcycle) == 0:
                    for i in range(agent_num):
                        state_collection[i] = np.r_[state_collection[i], [cur_state[i]]] if t else [cur_state[i]]
                        actions_collection[i] = np.r_[actions_collection[i], [virtual_actions[i]]] if t else [virtual_actions[i]]
                        value_collection[i] = np.r_[value_collection[i], v_t[i]] if t else [v_t[i]]
                        logp_t_collection[i] = np.r_[logp_t_collection[i], logp_t[i]] if t else [logp_t[i]]
                        done_collection[i] = np.r_[done_collection[i], done] if t else [done]
                        reward_collection[i] = np.r_[reward_collection[i], reward[i]] if t else [reward[i]]

                        # Update the observation
                        cur_state[i] = new_state[i]

                        episodic_reward += reward[i]
                        episodic_agent_reward[i] += reward[i]

                        v_t[i] = ppornd_agent[i].get_action([cur_state[i]], sess)[1][0]
                        value_collection[i] = np.r_[value_collection[i], v_t[i]]
                        next_values[i] = np.copy(value_collection[i][1:])
                        value_collection[i] = value_collection[i][:-1]
                        adv[i], target[i] = ppornd_agent[i].get_gaes(reward_collection[i], done_collection[i], value_collection[i], next_values[i], True)
                        ppornd_agent[i].update(state_collection[i], actions_collection[i], target[i], adv[i], logp_t_collection[i], sess)

            if done:
                break

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        ma40_reward = np.mean(ep_reward_list[-40:])
        ma1_reward = np.mean(ep_reward_list[-1:])
        print("Episode * {} * Avg Reward is ==> {}".format(trial, ma40_reward))
        print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        ma40_reward_list.append(ma40_reward)

        sess.run(total_reward.assign(ma1_reward))  # update accuracy variable
        train_summary_writer.add_summary(sess.run(total_reward_summary), trial)  # add summary

        for i in range(agent_num):
            v_t[i] = ppornd_agent[i].get_action([cur_state[i]], sess)[1][0]
            value_collection[i] = np.r_[value_collection[i], v_t[i]]
            next_values[i] = np.copy(value_collection[i][1:])
            value_collection[i] = value_collection[i][:-1]
            adv[i], target[i] = ppornd_agent[i].get_gaes(reward_collection[i], done_collection[i], value_collection[i], next_values[i], True)
            ppornd_agent[i].update(state_collection[i], actions_collection[i], target[i], adv[i], logp_t_collection[i], sess)
            ep_agent_reward_list[i].append(episodic_agent_reward[i])

            sess.run(agent_reward1[i].assign(np.mean(ep_agent_reward_list[i][-1:])))  # update accuracy variable
            sess.run(agent_reward40[i].assign(np.mean(ep_agent_reward_list[i][-40:])))  # update accuracy variable

            train_summary_writer.add_summary(sess.run(agent_reward1_summary[i]), trial)  # add summary
            train_summary_writer.add_summary(sess.run(agent_reward40_summary[i]), trial)  # add summary

        train_summary_writer.flush()

        f = open(fn_train_epoch_total_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)

        f.write('{},{},{}\n'.format(trial, ma1_reward, ma40_reward))
        f.close()

        for i in range(agent_num):
            f = open(fn_train_epoch_tl_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                     newline=None,
                     closefd=True, opener=None)
            f.write('{},{},{},{}\n'.format(trial, agent_crossName[i], np.mean(ep_agent_reward_list[i][-1:]), np.mean(ep_agent_reward_list[i][-40:])))
            f.close()

        if trial % args.model_save_period == 0:
            fn = "{}/model/sappo/SAPPO-{}-trial".format(io_home, problem_var)
            saver.save(sess, fn, global_step=trial)

if __name__ == "__main__":
    dir_name_list = [ #f"{args.io_home}/model",
                     f"{args.io_home}/model/{args.method}",
                     f"{args.io_home}/logs",
                     # f"{args.io_home}/output",
                     f"{args.io_home}/output/simulate",
                     f"{args.io_home}/output/test",
                     f"{args.io_home}/output/train",
                     #f"{args.io_home}/data/envs/salt/data",
    ]
    makeDirectories(dir_name_list)

    if args.mode == 'train':
        if args.method == 'sappo':
            run_sappo()
        elif args.method == 'ddqn':
            run_ddqn()
        elif args.method == 'ppornd':
            run_ppornd()
        elif args.method == 'ppoea':
            run_ppoea()
    elif args.mode == 'test':
        if args.method == 'sappo':
            sappo_test(args, args.model_num, problem_var)
        elif args.method == 'ddqn':
            ddqn_test(args, args.model_num, problem_var)
        elif args.method == 'ppornd':
            ppornd_test(args, args.model_num, problem_var)
        elif args.method == 'ppoea':
            sappo_test(args, args.model_num, problem_var)
    elif args.mode == 'simulate':
        ft_simulate(args)
