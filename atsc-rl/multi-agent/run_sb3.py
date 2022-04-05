import argparse

from stable_baselines3 import PPO, HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.cmd_util import make_vec_env

from env.sb3_env_gro import SALT_SAPPO_green_offset_single

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'simulate'], default='train',
                        help='train - RL model training, test - trained model testing, simulate - fixed-time simulation before test')
    parser.add_argument('--model-num', type=str, default='0',
                        help='trained model number for test mode')

    parser.add_argument('--method', choices=['sappo', 'ddqn', 'ppornd', 'ppoea', 'sac'], default='sappo',
                        help='')

    parser.add_argument('--map', choices=['dj_all', 'doan', 'sa_1_6_17'], default='sa_1_6_17')

    parser.add_argument('--target-TL', type=str, default="SA 1,SA 6,SA 17",
                        help="concatenate signal group with comma(ex. --target-TL SA 101,SA 104)")
    # parser.add_argument('--target-TL', type=str, default="SA 6",
    #                     help="concatenate signal group with comma(ex. --targetTL SA 101,SA 104)")
    parser.add_argument('--start-time', type=int, default=25400)
    parser.add_argument('--end-time', type=int, default=32400)

    # parser.add_argument('--result-comp', type=bool, default=True)
    parser.add_argument("--result-comp", type=str2bool, default="TRUE")

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
    parser.add_argument("--printOut", type=str2bool, default="TRUE", help='print result each step')

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
    parser.add_argument('--c-lr', type=float, default=0.005)
    parser.add_argument('--cp', type=float, default=0.0, help='action change penalty')
    parser.add_argument('--mmp', type=float, default=1.0, help='min max penalty')
    parser.add_argument('--actionp', type=float, default=0.2, help='action 0 or 1 prob.(-1~1): Higher value_collection select more zeros')

    ### PPO RND
    parser.add_argument('--gamma-i', type=float, default=0.11)

    ### PPO + RESNET
    parser.add_argument("--res", type=str2bool, default="TRUE")

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

    return args

args = getArgs()
trail_len = args.end_time - args.start_time

env = SALT_SAPPO_green_offset_single(args)
n_sampled_goal = 4
if args.method == 'sac':
    model = SAC(
        "MlpPolicy",
        env,
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

model.learn(int(100))
model.save(f'{args.method}_GRO_SINGLE_{args.target_TL}')

model.load(f'{args.method}_GRO_SINGLE_{args.target_TL}')

obs = env.reset()
for i in range(trail_len):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()