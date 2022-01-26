import argparse

import numpy as np
import matplotlib.pyplot as plt
import time


from policy.ddqn import DDQN
from policy.ppo import PPOAgent
from test import ddqn_test, sappo_test, ft_simulate

import keras.backend as K

from config import TRAIN_CONFIG

IS_DOCKERIZE = TRAIN_CONFIG['IS_DOCKERIZE']

if IS_DOCKERIZE:
    from env.salt_PennStateAction import SALT_doan_multi_PSA, getScenarioRelatedBeginEndTime
    from env.sappo_noConst import SALT_SAPPO_noConst, getScenarioRelatedBeginEndTime
    from env.sappo_offset import SALT_SAPPO_offset, getScenarioRelatedBeginEndTime
else:
    from env.salt_PennStateAction import SALT_doan_multi_PSA
    from env.sappo_noConst import SALT_SAPPO_noConst
    from env.sappo_offset import SALT_SAPPO_offset

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test', 'simulate'], default='train')
parser.add_argument('--model-num', type=str, default='260')

if IS_DOCKERIZE:
    parser.add_argument('--result-comp', type=bool, default=False)

    parser.add_argument('--start-time', type=int, default=0)
    parser.add_argument('--end-time', type=int, default=7200)
else:
    parser.add_argument('--resultComp', type=bool, default=False)

    parser.add_argument('--trainStartTime', type=int, default=0)
    parser.add_argument('--trainEndTime', type=int, default=7200)
    parser.add_argument('--testStartTime', type=int, default=0)
    parser.add_argument('--testEndTime', type=int, default=7200)

parser.add_argument('--epoch', type=int, default=3000)
parser.add_argument('--model-save-period', type=int, default=20)
parser.add_argument('--logprint', type=bool, default=False)
parser.add_argument('--printOut', type=bool, default=True, help='print result each step')

if IS_DOCKERIZE:
    parser.add_argument('--target-TL', type=str, default="SA 101,SA 104,SA 107,SA 111",
                        help="concatenate signal group with comma(ex. --target-TL SA 101,SA 104)")
else:
    parser.add_argument('--targetTL', type=str, default="SA 101,SA 104,SA 107,SA 111",
                        help="concatenate signal group with comma(ex. --targetTL SA 101,SA 104)")

parser.add_argument('--reward-func', choices=['pn', 'wt', 'wt_max', 'wq', 'wt_SBV', 'wt_SBV_max', 'wt_ABV'], default='wq',
                    help='pn - passed num, wt - wating time, wq - waiting q length')

parser.add_argument('--state', choices=['v', 'd', 'vd', 'vdd'], default='vd',
                    help='v - volume, d - density, vd - volume + density, vdd - volume / density')

parser.add_argument('--method', choices=['sappo', 'ddqn'], default='sappo',
                    help='')
parser.add_argument('--action', choices=['ps', 'kc', 'pss', 'o'], default='offset',
                    help='ps - phase selection(no constraints), kc - keep or change(limit phase sequence), '
                         'pss - phase-set selection, o - offset')

if IS_DOCKERIZE:
    parser.add_argument('--io-home', type=str, default='io')
    parser.add_argument('--scenario-file-path', type=str, default='io/data/sample/sample.json')

parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--action-t', type=int, default=12)

### PPO args
parser.add_argument('--tpi', type=int, default=1, help="train policy iteration")
parser.add_argument('--tvi', type=int, default=1, help="train value iteration")
parser.add_argument('--ppoEpoch', type=int, default=10)
parser.add_argument('--ppo_eps', type=float, default=0.1)
parser.add_argument('--_lambda', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cp', type=float, default=0.0, help='action change penalty')
parser.add_argument('--mmp', type=float, default=1.0, help='min max penalty')
parser.add_argument('--actionp', type=float, default=0.2, help='action 0 or 1 prob.(-1~1): Higher values select more zeros')

args = parser.parse_args()

problem_var = ""
# problem_var = "tau{}".format(args.tau)
# problem_var = "gamma{}".format(args.gamma)
# problem_var += "_yp0_actionT_{}".format(args.action_t)
problem_var += "_method_{}".format(args.method)
problem_var += "_state_{}".format(args.state)
problem_var += "_reward_{}".format(args.reward_func)
problem_var += "_action_{}".format(args.action)
problem_var += "_netsize_{}".format(TRAIN_CONFIG['network_size'])

if IS_DOCKERIZE:
    io_home = args.io_home
    output_train_dir = '{}/output/train'.format(io_home)
    fn_train_epoch_total_reward = "{}/train_epoch_total_reward.txt".format(output_train_dir)
    fn_train_epoch_tl_reward = "{}/train_epoch_tl_reward.txt".format(output_train_dir)

if IS_DOCKERIZE:
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
    if IS_DOCKERIZE:
        # trial_len = args.end_time - args.start_time
        scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
        start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
        end_time = args.end_time if args.end_time < scenario_end else scenario_end

        trial_len = end_time - start_time

    else:
        trial_len = args.trainEndTime - args.trainStartTime

    lr_update_period = TRAIN_CONFIG['lr_update_period']
    lr_update_decay = TRAIN_CONFIG['lr_update_decay']

    state_weight = env.state_weight

    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))

    if IS_DOCKERIZE:
        train_summary_writer = tf.summary.create_file_writer('{}/logs/{}/{}'.format(io_home, problem, time_data))
    else:
        train_summary_writer = tf.summary.create_file_writer('logs/{}/{}'.format(problem, time_data))

    if IS_DOCKERIZE:
        f = open(fn_train_epoch_total_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)
    else:
        f = open("output/train/train_epoch_total_reward.txt", mode='w+', buffering=-1, encoding='utf-8',
                 errors=None,
                 newline=None,
                 closefd=True, opener=None)

    f.write('epoch,reward,40ep_reward\n')
    f.close()

    if IS_DOCKERIZE:
        f = open(fn_train_epoch_tl_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)
    else:
        f = open("output/train/train_epoch_tl_reward.txt", mode='w+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)

    f.write('epoch,tl_name,reward,40ep_reward\n')
    f.close()

    agent_num = env.agent_num

    # updateTargetNetwork = 1000
    dqn_agent = []
    state_space_arr = []
    ep_agent_reward_list = []
    agent_crossName = []
    for i in range(agent_num):
        target_tl = list(env.target_tl_obj.keys())[i]
        state_space = env.target_tl_obj[target_tl]['state_space']
        state_space_arr.append(state_space)
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
            # print(new_state, reward)

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
            # print(episodic_agent_reward)

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        avg_reward1 = np.mean(ep_reward_list[-1:])
        print("Episode * {} * Avg Reward is ==> {}".format(trial, avg_reward))
        print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        avg_reward_list.append(avg_reward)

        with train_summary_writer.as_default():
            tf.summary.scalar('train/reward_40ep_mean', avg_reward, step=trial)
            tf.summary.scalar('train/reward', avg_reward1, step=trial)
            tf.summary.scalar('train/lr', dqn_agent[0].model.optimizer.lr.numpy(), step=trial)
            for i in range(agent_num):
                ep_agent_reward_list[i].append(episodic_agent_reward[i])
                tf.summary.scalar('train_agent_reward_40ep_mean/agent_{}_{}'.format(i,agent_crossName[i]), np.mean(ep_agent_reward_list[i][-40:]), step=trial)
                tf.summary.scalar('train_agent_reward/agent_{}_{}'.format(i,agent_crossName[i]), np.mean(ep_agent_reward_list[i][-1:]), step=trial)
            # tf.summary.scalar('train/epsilon', dqn_agent[0].epsilon, step=trial)

        if IS_DOCKERIZE :
            f = open(fn_train_epoch_total_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                     newline=None,
                     closefd=True, opener=None)
        else:
            f = open("output/train/train_epoch_total_reward.txt", mode='a+', buffering=-1, encoding='utf-8',
                     errors=None,
                     newline=None,
                     closefd=True, opener=None)

        f.write('{},{},{}\n'.format(trial, avg_reward1, avg_reward))
        f.close()

        for i in range(agent_num):
            if IS_DOCKERIZE:
                f = open(fn_train_epoch_tl_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                         newline=None,
                         closefd=True, opener=None)
            else:
                f = open("output/train/train_epoch_tl_reward.txt", mode='a+', buffering=-1, encoding='utf-8',
                         errors=None,
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
                if IS_DOCKERIZE:
                    # dqn_agent[i].save_model("{}/model/ddqn/PSA-{}-agent{}-trial-{}.model".format(io_home, problem_var, i, trial))
                    dqn_agent[i].save_model("{}/model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(io_home, problem_var, i, trial))
                else:
                    dqn_agent[i].save_model("model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(problem_var, i, trial))

def run_sappo():
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    if args.action=='kc':
        args.problem = "SAPPO_NoConstraints_" + problem_var
        env = SALT_SAPPO_noConst(args)
    if args.action=='offset':
        args.problem = "SAPPO_offset_" + problem_var
        env = SALT_SAPPO_offset(args)

    trials = args.epoch
    if IS_DOCKERIZE:
        # trial_len = args.end_time - args.start_time
        scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
        start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
        end_time = args.end_time if args.end_time < scenario_end else scenario_end
        trial_len = end_time - start_time
    else:
        trial_len = args.trainEndTime - args.trainStartTime

    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    # self.train_summary_writer = tf.summary.FileWriter('logs/SAPPO/{}/{}_{}'.format(args.problem, self.time_data, agentID))
    # train_summary_writer = tf.summary.FileWriter('logs/SAPPO/{}/{}'.format(args.problem, time_data))

    if IS_DOCKERIZE:
        train_summary_writer = tf.summary.FileWriter('{}/logs/SAPPO/{}/{}'.format(io_home, args.problem, time_data))
    else:
        train_summary_writer = tf.summary.FileWriter('logs/SAPPO/{}/{}'.format(args.problem, time_data))

    if IS_DOCKERIZE:
        f = open(fn_train_epoch_total_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)
    else:
        f = open("output/train/train_epoch_total_reward.txt", mode='w+', buffering=-1, encoding='utf-8',
                 errors=None,
                 newline=None,
                 closefd=True, opener=None)


    f.write('epoch,reward,40ep_reward\n')
    f.close()

    if IS_DOCKERIZE:
        f = open(fn_train_epoch_tl_reward, mode='w+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)
    else:
        f = open("output/train/train_epoch_tl_reward.txt", mode='w+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)

    f.write('epoch,tl_name,reward,40ep_reward\n')
    f.close()

    agent_num = env.agent_num

    sappo_agent = []
    state_space_arr = []
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
        state_space_arr.append(state_space)
        action_space = env.sa_obj[target_sa]['action_space']
        action_min = env.sa_obj[target_sa]['action_min']
        action_max = env.sa_obj[target_sa]['action_max']
        print(f"{target_sa}, state space {state_space} action space {action_space}, action min {action_min}, action max {action_max}")
        sappo_agent.append(PPOAgent(args=args, state_space=state_space, action_space=action_space, action_min=action_min, action_max=action_max, agentID=i, fn=''))
        ep_agent_reward_list.append([])
        agent_crossName.append(env.sa_obj[target_sa]['crossName_list'])

        agent_reward1.append(tf.Variable(0, dtype=tf.float32))
        agent_reward40.append(tf.Variable(0, dtype=tf.float32))
        agent_reward1_summary.append(tf.summary.scalar('train_agent_reward/agent_{}'.format(list(env.sa_obj.keys())[i]), agent_reward1[i]))  # summary to write to TensorBoard
        agent_reward40_summary.append(tf.summary.scalar('train_agent_reward_40ep_mean/agent_{}'.format(list(env.sa_obj.keys())[i]), agent_reward40[i]))  # summary to write to TensorBoard

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(sappo_agent)

    print("env.sa_obj", env.sa_obj)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    for trial in range(trials):
        actions = []
        logits = []
        value_t = []
        logprobability_t = []

        actionss = []
        states = []
        values = []
        logp_ts = []
        dones = []
        rewards = []

        v_t = []
        next_values = []
        adv = []
        target = []

        sa_cycle = []

        for target_sa in env.sa_obj:
            actions.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            logits.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            value_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            logprobability_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])

            actionss.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            states.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            values.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            logp_ts.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            dones.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
            rewards.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])

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
            # print("range(agent_num)", range(agent_num))
            for i in range(agent_num):
                actions[i], value_t[i], logprobability_t[i] = sappo_agent[i].get_action([cur_state[i]], sess)
                actions[i], value_t[i], logprobability_t[i] = actions[i][0], value_t[i][0], logprobability_t[i][0]

                target_sa = list(env.sa_obj.keys())[i]
                discrete_action = []
                for di in range(len(actions[i])):
                    # discrete_action.append(np.digitize(actions[i][di], bins=env.sa_obj[target_sa]['duration_bins_list'][di]))
                    if args.action=='kc':
                        discrete_action.append(0 if actions[i][di] < args.actionp else 1)
                    if args.action=='offset':
                        discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2))

                discrete_actions.append(discrete_action)
            new_state, reward, done, _ = env.step(discrete_actions)

            for i in range(agent_num):
                states[i] = np.r_[states[i], [cur_state[i]]] if t else [cur_state[i]]
                actionss[i] = np.r_[actionss[i], [actions[i]]] if t else [actions[i]]
                values[i] = np.r_[values[i], value_t[i]] if t else [value_t[i]]
                logp_ts[i] = np.r_[logp_ts[i], logprobability_t[i]] if t else [logprobability_t[i]]
                dones[i] = np.r_[dones[i], done] if t else [done]
                rewards[i] = np.r_[rewards[i], reward[i]] if t else [reward[i]]

                # Update the observation
                cur_state[i] = new_state[i]

                episodic_reward += reward[i]
                episodic_agent_reward[i] += reward[i]

            if done:
                break

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        avg_reward1 = np.mean(ep_reward_list[-1:])
        print("Episode * {} * Avg Reward is ==> {}".format(trial, avg_reward))
        print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        avg_reward_list.append(avg_reward)

        sess.run(total_reward.assign(avg_reward1))  # update accuracy variable
        train_summary_writer.add_summary(sess.run(total_reward_summary), trial)  # add summary

        for i in range(agent_num):
            v_t[i] = sappo_agent[i].get_action([cur_state[i]], sess)[1][0]
            values[i] = np.r_[values[i], v_t[i]]
            next_values[i] = np.copy(values[i][1:])
            values[i] = values[i][:-1]
            adv[i], target[i] = sappo_agent[i].get_gaes(rewards[i], dones[i], values[i], next_values[i], True)
            sappo_agent[i].update(states[i], actionss[i], target[i], adv[i], logp_ts[i], sess)
            ep_agent_reward_list[i].append(episodic_agent_reward[i])

            sess.run(agent_reward1[i].assign(np.mean(ep_agent_reward_list[i][-1:])))  # update accuracy variable
            sess.run(agent_reward40[i].assign(np.mean(ep_agent_reward_list[i][-40:])))  # update accuracy variable

            train_summary_writer.add_summary(sess.run(agent_reward1_summary[i]), trial)  # add summary
            train_summary_writer.add_summary(sess.run(agent_reward40_summary[i]), trial)  # add summary

        train_summary_writer.flush()

        if IS_DOCKERIZE :
            f = open(fn_train_epoch_total_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                     newline=None,
                     closefd=True, opener=None)
        else:
            f = open("output/train/train_epoch_total_reward.txt", mode='a+', buffering=-1, encoding='utf-8',
                     errors=None,
                     newline=None,
                     closefd=True, opener=None)

        f.write('{},{},{}\n'.format(trial, avg_reward1, avg_reward))
        f.close()

        for i in range(agent_num):
            if IS_DOCKERIZE:
                f = open(fn_train_epoch_tl_reward, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                         newline=None,
                         closefd=True, opener=None)
            else:
                f = open("output/train/train_epoch_tl_reward.txt", mode='a+', buffering=-1, encoding='utf-8',
                         errors=None,
                         newline=None,
                         closefd=True, opener=None)

            f.write('{},{},{},{}\n'.format(trial, agent_crossName[i], np.mean(ep_agent_reward_list[i][-1:]), np.mean(ep_agent_reward_list[i][-40:])))
            f.close()

        if trial % 20 == 0:
            fn = "model/ppo/SAPPO-{}-trial".format(problem_var)
            saver.save(sess, fn, global_step=trial)

        if trial % args.model_save_period == 0:
            for i in range(agent_num):
                if IS_DOCKERIZE:
                    fn = "{}/model/ppo/SAPPO-{}-trial".format(io_home, problem_var)
                    saver.save(sess, fn, global_step=trial)
                else:
                    fn = "model/ppo/SAPPO-{}-trial".format(problem_var)
                    saver.save(sess, fn, global_step=trial)



if __name__ == "__main__":
    if IS_DOCKERIZE:
        if args.method == 'sappo':
            dir_name_list = [ #f"{args.io_home}/model",
                             f"{args.io_home}/model/sappo",
                             f"{args.io_home}/logs",
                             # f"{args.io_home}/output",
                             f"{args.io_home}/output/ft",
                             f"{args.io_home}/output/rl",
                             f"{args.io_home}/output/test",
                             f"{args.io_home}/output/train",
                             #f"{args.io_home}/data/envs/salt/data",
            ]
        elif args.method =='ddqn':
            dir_name_list = [  # f"{args.io_home}/model",
                f"{args.io_home}/model/ddqn",
                f"{args.io_home}/logs",
                # f"{args.io_home}/output",
                f"{args.io_home}/output/ft",
                f"{args.io_home}/output/rl",
                f"{args.io_home}/output/test",
                f"{args.io_home}/output/train",
                # f"{args.io_home}/data/envs/salt/data",
            ]

        makeDirectories(dir_name_list)

    if args.mode == 'train':
        if args.method == 'sappo':
            run_sappo()
        elif args.method == 'ddqn':
            run_ddqn()
    elif args.mode == 'test':
        if args.method == 'sappo':
            sappo_test(args, args.model_num, problem_var)
        elif args.method == 'ddqn':
            ddqn_test(args, args.model_num, problem_var)
    elif args.mode == 'simulate':
        ft_simulate(args)
