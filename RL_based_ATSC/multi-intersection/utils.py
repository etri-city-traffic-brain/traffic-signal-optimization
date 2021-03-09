import os,json
import torch
from configs import EXP_CONFIGS
def save_params(configs, time_data):
    with open(os.path.join(configs['current_path'], 'training_data', '{}.json'.format(time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)


def load_params(configs, file_name):
    ''' replay_name from flags.replay_name '''
    with open(os.path.join(configs['current_path'], 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs


def update_tensorboard(writer,epoch,env,agent,arrived_vehicles):
    env.update_tensorboard(writer,epoch)
    agent.update_tensorboard(writer,epoch)
    writer.add_scalar('episode/arrived_num', arrived_vehicles,
                        EXP_CONFIGS['max_steps']*epoch)  # 1 epoch마다

    writer.flush()


interest_list = [
    {
        'id': 'u_1_1',
        'inflow': 'n_1_0_to_n_1_1',
        'outflow': 'n_1_1_to_n_1_2',
    },
    {
        'id': 'r_1_1',
        'inflow': 'n_2_1_to_n_1_1',
        'outflow': 'n_1_1_to_n_0_1',
    },
    {
        'id': 'd_1_1',
        'inflow': 'n_1_2_to_n_1_1',
        'outflow': 'n_1_1_to_n_1_0',
    },
    {
        'id': 'l_1_1',
        'inflow': 'n_0_1_to_n_1_1',
        'outflow': 'n_1_1_to_n_2_1',
    }
]
