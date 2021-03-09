import torch
import traci
import time
from utils import load_params
from Agent.base import merge_dict_non_conflict


def city_dqn_test(flags, sumoCmd, configs):
    # Environment Setting
    from Agent.super_dqn import Trainer
    from Env.CityEnv import CityEnv
    # init test setting
    if flags.replay_name is not None:
        # 여기앞에 configs 설정해도 의미 없음
        configs = load_params(configs, flags.replay_name)
        configs['replay_epoch'] = str(flags.replay_epoch)
        configs['mode'] = 'test'

    phase_num_matrix = torch.tensor(  # 각 tl이 갖는 최대 phase갯수
        [len(configs['traffic_node_info'][index]['phase_duration']) for _, index in enumerate(configs['traffic_node_info'])])

    agent = Trainer(configs)
    agent.save_params(configs['time_data'])
    agent.load_weights(flags.replay_name)
    # init training
    NUM_AGENT = configs['num_agent']
    TL_RL_LIST = configs['tl_rl_list']
    MAX_PHASES = configs['max_phase_num']
    MAX_STEPS = configs['max_steps']
    OFFSET = torch.tensor(configs['offset'],  # i*10
                          device=configs['device'], dtype=torch.int)
    TL_PERIOD = torch.tensor(
        configs['tl_period'], device=configs['device'], dtype=torch.int)
    # state initialization
    # agent setting
    # check performance
    avg_waiting_time = 0
    avg_part_velocity = 0
    avg_velocity = 0
    arrived_vehicles = 0
    part_velocity = list()
    waiting_time = list()
    total_velocity = list()
    # travel time
    travel_time = list()
    with torch.no_grad():
        step = 0
        traci.start(sumoCmd)
        env = CityEnv(configs)
        # Total Initialization
        actions = torch.zeros(
            (NUM_AGENT, configs['action_size']), dtype=torch.int, device=configs['device'])
        # Mask Matrix : TL_Period가 끝나면 True
        mask_matrix = torch.ones(
            (NUM_AGENT), dtype=torch.bool, device=configs['device'])

        # MAX Period까지만 증가하는 t
        t_agent = torch.zeros(
            (NUM_AGENT), dtype=torch.int, device=configs['device'])
        t_agent -= OFFSET

        # Action configs['offset']on Matrix : 비교해서 동일할 때 collect_state, 없는 state는 zero padding
        action_matrix = torch.zeros(
            (NUM_AGENT, MAX_PHASES), dtype=torch.int, device=configs['device'])  # 노란불 3초 해줘야됨
        action_index_matrix = torch.zeros(
            (NUM_AGENT), dtype=torch.long, device=configs['device'])  # 현재 몇번째 phase인지
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
            # action형태로 변환 # 다음으로 넘어가야할 시점에 대한 matrix
            action_matrix = env.calc_action(
                action_matrix, actions, mask_matrix)
            # 누적값으로 나타남

            # environment에 적용
            # action 적용함수, traci.simulationStep 있음
            next_state = env.step(
                actions, mask_matrix, action_index_matrix, action_update_mask)

            # 전체 1초증가 # traci는 env.step에
            step += 1
            t_agent += 1
            # 최대에 도달하면 0으로 초기화 (offset과 비교)
            clear_matrix = torch.eq(t_agent % TL_PERIOD, 0)
            t_agent[clear_matrix] = 0


            # 넘어가야된다면 action index증가 (by tensor slicing)
            action_update_mask = torch.eq(  # update는 단순히 진짜 현시만 받아서 결정해야됨
                t_agent, action_matrix[0, action_index_matrix]).view(NUM_AGENT)  # 0,인 이유는 인덱싱
            action_index_matrix[action_update_mask] += 1
            # agent의 최대 phase를 넘어가면 해당 agent의 action index 0으로 초기화
            action_index_matrix[clear_matrix] = 0

            # mask update, matrix True로 전환
            mask_matrix[clear_matrix] = True
            mask_matrix[~clear_matrix] = False


            # check performance
            for _, interests in enumerate(configs['interest_list']):
                # delete 중복
                dup_list = list()
                for interest in interests:
                    inflow = interest['inflow']
                    outflow = interest['outflow']
                    # 신호군 흐름
                    if inflow != None and inflow not in dup_list:
                        # 차량의 대기시간, 차량이 있을 때만
                        if traci.edge.getLastStepVehicleNumber(inflow) != 0:
                            waiting_time.append(traci.edge.getWaitingTime(inflow))#/float(
                                #traci.edge.getLastStepVehicleNumber(inflow)))
                            # 차량의 평균속도
                            part_velocity.append(
                                traci.edge.getLastStepMeanSpeed(inflow))
                            tmp_travel = traci.edge.getTraveltime(inflow)
                            if tmp_travel <= 320:  # 이상한 값 거르기
                                travel_time.append(tmp_travel)
                        dup_list.append(inflow)

                    if outflow != None and outflow not in dup_list:
                        if traci.edge.getLastStepVehicleNumber(outflow) != 0:
                            part_velocity.append(
                                traci.edge.getLastStepMeanSpeed(interest['outflow']))
                        dup_list.append(interest['outflow'])
            # edge_list=traci.edge.getIDList()
            # for edgeid in edge_list:
            #     if traci.edge.getLastStepVehicleNumber(edgeid) !=None:
            #         total_velocity.append(traci.edge.getLastStepMeanSpeed(edgeid))
            state = next_state
            # info
            
            arrived_vehicles += traci.simulation.getArrivedNumber()

            state = next_state

        b = time.time()
        traci.close()
        print("time:", b-a)
        avg_part_velocity = torch.tensor(
            part_velocity, dtype=torch.float).mean()
        avg_velocity = torch.tensor(total_velocity, dtype=torch.float).mean()
        avg_part_velocity = torch.tensor(
            part_velocity, dtype=torch.float).mean()
        avg_travel_time = torch.tensor(travel_time, dtype=torch.float).mean()
        avg_waiting_time = torch.tensor(waiting_time, dtype=torch.float).mean()
        print('======== arrived number:{} avg waiting time:{},avg velocity:{} avg_part_velocity: {} avg_travel_time: {}'.format(
            arrived_vehicles, avg_waiting_time, avg_velocity, avg_part_velocity, avg_travel_time))
