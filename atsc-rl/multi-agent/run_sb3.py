import argparse

from stable_baselines3 import PPO, HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.cmd_util import make_vec_env

from config import TRAIN_CONFIG

IS_DOCKERIZE = TRAIN_CONFIG['IS_DOCKERIZE']

from env.sb3_env_gro import SALT_SAPPO_green_offset_single

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test', 'simulate'], default='train')
parser.add_argument('--model-num', type=str, default='600')

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
parser.add_argument('--warmupTime', type=int, default=600)
parser.add_argument('--model-save-period', type=int, default=20)
parser.add_argument('--logprint', type=bool, default=False)
parser.add_argument('--printOut', type=bool, default=True, help='print result each step')

if IS_DOCKERIZE:
    parser.add_argument('--target-TL', type=str, default="SA 101,SA 104,SA 107,SA 111",
                        help="concatenate signal group with comma(ex. --target-TL SA 101,SA 104)")
else:
    # parser.add_argument('--target-TL', type=str, default="SA 101,SA 104,SA 107,SA 111",
    #                     help="concatenate signal group with comma(ex. --targetTL SA 101,SA 104)")
    parser.add_argument('--target-TL', type=str, default="SA 6",
                        help="concatenate signal group with comma(ex. --targetTL SA 101,SA 104)")

parser.add_argument('--reward-func', choices=['pn', 'wt', 'wt_max', 'wq', 'wq_median', 'wq_min', 'wq_max', 'wt_SBV', 'wt_SBV_max', 'wt_ABV', 'tt', 'cwq'], default='cwq',
                    help='pn - passed num, wt - wating time, wq - waiting q length, tt - travel time, cwq - cumulative waiting q length')

parser.add_argument('--state', choices=['v', 'd', 'vd', 'vdd'], default='vdd',
                    help='v - volume, d - density, vd - volume + density, vdd - volume / density')

parser.add_argument('--method', choices=['sappo', 'ddqn', 'ppornd', 'ppoea', 'ppo', 'sac'], default='sac',
                    help='')
parser.add_argument('--action', choices=['ps', 'kc', 'pss', 'o', 'gr', 'gro'], default='gro',
                    help='ps - phase selection(no constraints), kc - keep or change(limit phase sequence), '
                         'pss - phase-set selection, o - offset, gr - green ratio, gro - green ratio+offset')
parser.add_argument('--map', choices=['dj', 'doan'], default='dj',
                    help='dj - Daejeon all region, doan - doan 111 tss')

if IS_DOCKERIZE:
    parser.add_argument('--io-home', type=str, default='io')
    parser.add_argument('--scenario-file-path', type=str, default='io/data/sample/sample.json')


parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gamma-i', type=float, default=0.11)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--action-t', type=int, default=12)
parser.add_argument('--offsetrange', type=int, default=2, help="offset side range")

### PPO args
parser.add_argument('--tpi', type=int, default=1, help="train policy iteration")
parser.add_argument('--tvi', type=int, default=1, help="train value iteration")
parser.add_argument('--ppoEpoch', type=int, default=10)
parser.add_argument('--ppo_eps', type=float, default=0.1)
parser.add_argument('--_lambda', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--cp', type=float, default=0.0, help='action change penalty')
parser.add_argument('--mmp', type=float, default=1.0, help='min max penalty')
parser.add_argument('--actionp', type=float, default=0.2, help='action 0 or 1 prob.(-1~1): Higher values select more zeros')
parser.add_argument('--controlcycle', type=int, default=5)
parser.add_argument('--res', type=bool, default=True)
parser.add_argument('--logstdI', type=float, default=0.5)

### GREEN RATIO args
parser.add_argument('--addTime', type=int, default=2)

args = parser.parse_args()

if args.map == 'dj':
    args.trainStartTime = 25200
    args.trainEndTime = 32400
    # args.trainEndTime = 32400
    args.testStartTime = 25200
    args.testEndTime = 32400

env = SALT_SAPPO_green_offset_single(args)
n_sampled_goal = 4
# model = PPO("MlpPolicy", env, verbose=1)
# model = SAC('MlpPolicy', env, train_freq=1, gradient_steps=2, verbose=1)
if args.method == 'sac':
    model = SAC(
        "MlpPolicy",
        env,
        # replay_buffer_class=HerReplayBuffer,
        # replay_buffer_kwargs=dict(
        #   n_sampled_goal=n_sampled_goal,
        #   goal_selection_strategy="future",
        #   # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
        #   # we have to manually specify the max number of steps per episode
        #   max_episode_length=100,
        #   online_sampling=True,
        # ),
        verbose=1,
        buffer_size=int(2000),
        learning_rate=0.005,
        gamma=0.99,
        batch_size=4,
        policy_kwargs=dict(net_arch=[512, 512, 512, 512]),
    )
elif args.method == 'ppo':
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        gamma=0.999,
        batch_size=8,
        policy_kwargs=dict(net_arch=[512, 512, 512, 512, 512]),
    )

model.learn(int(8*1000))
model.save(f'{args.method}_GRO_SINGLE_{args.target_TL}')

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    if done:
        obs = env.reset()

env.close()