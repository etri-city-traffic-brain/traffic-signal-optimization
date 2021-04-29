import json
import os
from random import random
import sys
import time
import traci
import traci.constants as tc
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils import update_tensorboard
from Agent.base import merge_dict


def city_dqn_train(configs, time_data, sumoCmd):
    from Agent.super_dqn import Trainer
    if configs['model'] == 'city':
        from Env.CityEnv import CityEnv

    phase_num_matrix = torch.tensor(  # 각 tl이 갖는 최대 phase갯수
        [len(configs['traffic_node_info'][index]['phase_duration']) for _, index in enumerate(configs['traffic_node_info'])])
    # init agent and tensorboard writer
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    agent = Trainer(configs)
    # save hyper parameters
    agent.save_params(time_data)
    # init training
    NUM_AGENT = configs['num_agent']
    DEVICE = configs['device']
    TL_RL_LIST = configs['tl_rl_list']
    MAX_PHASES = configs['max_phase_num']
    MAX_STEPS = configs['max_steps']
    OFFSET = torch.tensor(configs['offset'],  # i*10
                          device=DEVICE, dtype=torch.int)
    TL_PERIOD = torch.tensor(
        configs['tl_period'], device=DEVICE, dtype=torch.int)
    epoch = 0
    while epoch < configs['num_epochs']:
        step = 0
        if configs['randomness'] == True:
            tmp_sumoCmd = sumoCmd+['--scale', str(1.5+random())]  # 1.5~2.5
        else:
            if configs['network'] == 'dunsan' or  'grid' in configs['network']:
                tmp_sumoCmd = sumoCmd+['--scale', str(configs['scale'])]
            else:
                tmp_sumoCmd = sumoCmd
        traci.start(tmp_sumoCmd)
        env = CityEnv(configs)
        # Total Initialization
        actions = torch.zeros(
            (NUM_AGENT, configs['action_size']), dtype=torch.int, device=DEVICE)
        # Mask Matrix : TL_Period가 끝나면 True
        mask_matrix = torch.zeros(
            (NUM_AGENT), dtype=torch.bool, device=DEVICE)

        # MAX Period까지만 증가하는 t
        t_agent = torch.zeros(
            (NUM_AGENT), dtype=torch.int, device=DEVICE)
        t_agent -= OFFSET

        # Action configs['offset']on Matrix : 비교해서 동일할 때 collect_state, 없는 state는 zero padding
        action_matrix = torch.zeros(
            (NUM_AGENT, MAX_PHASES), dtype=torch.int, device=DEVICE)  # 노란불 3초 해줘야됨
        action_index_matrix = torch.zeros(
            (NUM_AGENT), dtype=torch.long, device=DEVICE)  # 현재 몇번째 phase인지
        action_update_mask = torch.eq(   # action이 지금 update해야되는지 확인
            t_agent, action_matrix[0, action_index_matrix]).view(NUM_AGENT)  # 0,인 이유는 인덱싱

        # 최대에 도달하면 0으로 초기화 (offset과 비교)
        clear_matrix = torch.eq(t_agent % TL_PERIOD, 0)
        t_agent[clear_matrix] = 0
        # action 넘어가야된다면 action index증가 (by tensor slicing)
        action_index_matrix[action_update_mask] += 1
        action_index_matrix[clear_matrix] = 0

        # mask update, matrix True로 전환
        mask_matrix[clear_matrix] = True
        mask_matrix[~clear_matrix] = False

        # state initialization
        state = env.collect_state(
            action_update_mask, action_index_matrix, mask_matrix)
        total_reward = 0

        # agent setting
        arrived_vehicles = 0
        a = time.time()
        while step < MAX_STEPS:
            # action 을 정하고
            actions = agent.get_action(state, mask_matrix)
            if mask_matrix.sum()>0:
                print(actions.transpose(1,2))
            # action형태로 변환 # 다음으로 넘어가야할 시점에 대한 matrix
            action_matrix = env.calc_action(
                action_matrix, actions, mask_matrix)
            # 누적값으로 나타남

            # environment에 적용
            # action 적용함수, traci.simulationStep 있음
            env.step(
                actions, mask_matrix, action_index_matrix, action_update_mask)

            # 전체 1초증가 # traci는 env.step에
            step += 1
            t_agent += 1
            # 최대에 도달하면 0으로 초기화 (offset과 비교)
            clear_matrix = torch.eq(t_agent % TL_PERIOD, 0)

            # action 넘어가야된다면 action index증가 (by tensor slicing)
            for idx,_ in enumerate(TL_RL_LIST):
                action_update_mask[idx] = torch.eq(  # update는 단순히 진짜 현시만 받아서 결정해야됨
                    t_agent[idx], action_matrix[idx, action_index_matrix[idx]].view(-1))  # 0,인 이유는 인덱싱

            action_index_matrix[action_update_mask] += 1
            # agent의 최대 phase를 넘어가면 해당 agent의 action index 0으로 초기화
            action_index_matrix[clear_matrix] = 0
            
            # mask update, matrix True로 전환
            t_agent[clear_matrix] = 0
            # print(t_agent,action_index_matrix,step,action_update_mask)
            mask_matrix[clear_matrix] = True
            mask_matrix[~clear_matrix] = False

            next_state = env.collect_state(
                action_update_mask, action_index_matrix, mask_matrix)
            # env속에 agent별 state를 꺼내옴, max_offset+period 이상일 때 시작
            if step >= int(torch.max(OFFSET)+torch.max(TL_PERIOD)) and mask_matrix.sum() > 0:
                rep_state, rep_action, rep_reward, rep_next_state = env.get_state(
                    mask_matrix)
                agent.save_replay(rep_state, rep_action, rep_reward,
                                  rep_next_state, mask_matrix)  # dqn
            # update
            agent.update(mask_matrix)

            state = next_state
            # info
            arrived_vehicles += traci.simulation.getArrivedNumber()

        agent.target_update(epoch)
        agent.update_hyperparams(epoch)  # lr and epsilon upate
        b = time.time()
        traci.close()
        print("time:", b-a)
        epoch += 1
        # once in an epoch
        print('======== {} epoch/ return: {:.5f} arrived number:{}'.format(epoch,
                                                                           env.cum_reward.sum(), arrived_vehicles))
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        env.test_val=0
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}'.format(epoch))

    writer.close()
