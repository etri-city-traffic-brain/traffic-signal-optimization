from typing import NewType
import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy


class Memory():
    def __init__(self, configs):
        self.configs = configs
        self.reward = torch.zeros(
            1, dtype=torch.float, device=configs['device'])
        self.state = torch.zeros(
            (1, self.configs['state_space'], 4, 1), dtype=torch.float, device=configs['device'])
        self.next_state = torch.zeros_like(self.state)
        self.action = torch.zeros(
            (1, self.configs['action_size']), dtype=torch.int, device=configs['device'])


class CityEnv(baseEnv):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.device = self.configs['device']
        self.phase_action_matrix = torch.zeros(  # 누적합산 이전의 action_matrix
            (self.configs['num_agent'], self.configs['max_phase_num']), dtype=torch.int, device=self.device)  # reward 계산시 사용
        self.tl_list = traci.trafficlight.getIDList()
        self.tl_rl_list = self.configs['tl_rl_list']
        self.num_agent = len(self.tl_rl_list)
        self.side_list = ['u', 'r', 'd', 'l']
        self.interest_list = self.configs['interest_list']
        self.node_interest_pair = self.configs['node_interest_pair']

        self.reward = torch.zeros(
            (1, self.num_agent), dtype=torch.float, device=self.configs['device'])
        self.cum_reward = torch.zeros_like(self.reward)
        self.state_space = self.configs['state_space']
        self.action_size = self.configs['action_size']
        self.traffic_node_info = self.configs['traffic_node_info']
        self.nodes = self.configs['node_info']

        self.before_action_update_mask = torch.zeros(
            self.num_agent, dtype=torch.long, device=self.device)
        self.before_action_index_matrix = torch.zeros(
            self.num_agent, dtype=torch.long, device=self.device)
        self.tl_rl_memory = list()
        for _ in range(self.num_agent):
            self.tl_rl_memory.append(Memory(self.configs))

        # action의 mapping을 위한 matrix
        self.min_phase = torch.tensor(
            self.configs['min_phase'], dtype=torch.int, device=self.device)
        self.max_phase = torch.tensor(
            self.configs['max_phase'], dtype=torch.int, device=self.device)
        self.common_phase = torch.tensor(
            self.configs['common_phase'], dtype=torch.int, device=self.device)
        self.matrix_actions = torch.tensor(
            self.configs['matrix_actions'], dtype=torch.int, device=self.device)
        # phase 갯수 list 생성
        self.num_phase_list = list()
        for phase in self.common_phase:
            self.num_phase_list.append(len(phase))

        self.left_lane_num_dict = dict()
        # lane 정보 저장
        for interest in self.node_interest_pair:
            # 모든 inflow에 대해서
            for pair in self.node_interest_pair[interest]:
                if pair['inflow'] == None:
                    continue
                self.left_lane_num_dict[pair['inflow']] = traci.edge.getLaneNumber(
                    pair['inflow'])-1
        
        # self.test_val=list()
        # for i in self.tl_rl_list:
        #     self.test_val.append(0)

    def get_state(self, mask):
        '''
        매 주기마다 매 주기 이전의 state, 현재 state, reward를 반환하는 함수
        reward,next_state<-state 초기화 해줘야됨
        '''

        state = torch.zeros(
            (1, self.state_space, 4, self.num_agent), dtype=torch.float, device=self.device)
        next_state = torch.zeros_like(state)
        action = torch.zeros(
            (1, self.num_agent, self.action_size), dtype=torch.int, device=self.device)
        reward = torch.zeros((1, self.num_agent),
                             dtype=torch.float, device=self.device)
        for index in torch.nonzero(mask):
            state[0, :, :, index] = deepcopy(self.tl_rl_memory[index].state)
            action[0, index, :] = deepcopy(self.tl_rl_memory[index].action)
            next_state[0, :, :, index] = deepcopy(
                self.tl_rl_memory[index].next_state)
            reward[0, index] = deepcopy(self.tl_rl_memory[index].reward)
            # mask index's reward clear
            self.tl_rl_memory[index].reward = 0

        return state, action, reward, next_state

    def collect_state(self, action_update_mask, action_index_matrix, mask_matrix):
        '''
        매초 마다 update할 것이 있는지 확인해야함
        전과 비교해서 인덱스가 늘어나고 그 인덱스가

        Max Pressure based control
        각 node에 대해서 inflow 차량 수와 outflow 차량수 + 해당 방향이라는 전제에서
        '''
        # Reward 저장을 위한 mask 생성
        action_change_mask = torch.zeros_like(action_update_mask)
        for index in torch.nonzero(action_update_mask):
            if action_index_matrix[index] in self.traffic_node_info[self.tl_rl_list[index]]['phase_index']:
                # action_index_matrix상의 값이 next state를 받아와야하는 index일 경우
                action_change_mask[index] = True
                # self.test_val[index]+=1

        # Reward
        for index in torch.nonzero(action_change_mask):
            # self.test_val+=1
            outflow = 0
            inflow = 0
            interests = self.node_interest_pair[self.tl_rl_list[index]]
            for interest in interests:
                if interest['outflow']:  # None이 아닐 때 행동
                    outflow += (traci.edge.getLastStepVehicleNumber(
                        interest['outflow']))/100.0
                if interest['inflow']:  # None이 아닐 때 행동
                    inflow += traci.edge.getLastStepVehicleNumber(
                        interest['inflow'])/100.0
            # pressure=inflow-outflow
            # reward cumulative sum
            pressure = torch.tensor(
                abs(inflow-outflow), dtype=torch.float, device=self.device)/100.0
            self.reward[0, index] -= pressure
            self.tl_rl_memory[index].reward -= pressure

        # penalty
        for index in torch.nonzero(mask_matrix):
            if self.phase_action_matrix[index].sum() != 0:
                phase_index = torch.tensor(
                    self.traffic_node_info[self.tl_rl_list[index]]['phase_index'], device=self.device).view(1, -1).long()
                # penalty for phase duration more than maxDuration
                if torch.gt(self.phase_action_matrix[index].gather(dim=1, index=phase_index), torch.tensor(self.traffic_node_info[self.tl_rl_list[index]]['max_phase'])).sum():
                    self.reward[0, index] -= 0.4
                    self.tl_rl_memory[index].reward -= 0.4  # penalty

                # penalty for phase duration less than minDuration
                if torch.gt(torch.tensor(self.traffic_node_info[self.tl_rl_list[index]]['min_phase']), self.phase_action_matrix[index].gather(dim=1, index=phase_index)).sum():
                    self.reward[0, index] -= 0.4
                    self.tl_rl_memory[index].reward -= 0.4  # penalty

        # action 변화를 위한 state
        next_states = torch.zeros(
            (1, self.state_space, 4, self.num_agent), dtype=torch.float, device=self.device)
        # print(action_update_mask)
        for idx in torch.nonzero(action_change_mask):
            next_state = list()
            # 모든 rl node에 대해서
            phase_type_tensor = torch.tensor(self.configs['phase_type'][idx])
            # vehicle state
            veh_state = torch.zeros(
                (self.state_space-2-2), dtype=torch.float, device=self.device)
            for j, pair in enumerate(self.node_interest_pair[self.tl_rl_list[idx]]):
                # 모든 inflow에 대해서
                if pair['inflow'] is None:
                    veh_state[j*2] = 0.0
                    veh_state[j*2+1] = 0.0
                else:
                    left_movement = traci.lane.getLastStepVehicleNumber(
                        pair['inflow']+'_{}'.format(self.left_lane_num_dict[pair['inflow']]))/100.0  # 멈춘애들 계산
                    # 직진
                    veh_state[j*2] = traci.edge.getLastStepVehicleNumber(
                        pair['inflow'])/100.0-left_movement  # 가장 좌측에 멈춘 친구를 왼쪽차선 이용자로 판단
                    # 좌회전
                    veh_state[j*2+1] = left_movement
            # duration 차이의 tensor
            min_dur_tensor = torch.tensor(
                self.traffic_node_info[self.tl_rl_list[idx]]['dif_min'][int(action_index_matrix[idx]/2)], dtype=torch.float, device=self.device).view(-1)
            max_dur_tensor = torch.tensor(
                self.traffic_node_info[self.tl_rl_list[idx]]['dif_max'][int(action_index_matrix[idx]/2)], dtype=torch.float, device=self.device).view(-1)
            next_state = torch.cat((veh_state, phase_type_tensor, min_dur_tensor, max_dur_tensor), dim=0).view(
                self.state_space, 1)
            # print(next_state,idx,self.configs['phase_type'][idx])
            # print(next_state)

            self.tl_rl_memory[idx].state[:, :, (action_index_matrix[idx]/2).long()
                                         ] = self.tl_rl_memory[idx].next_state[:, :, (action_index_matrix[idx]/2).long()].detach().clone()
            self.tl_rl_memory[idx].next_state[:, :, (action_index_matrix[idx]/2).long()
                                              ] = next_state.view(1, self.state_space, 1, 1).detach().clone()

        for idx in torch.nonzero(mask_matrix):
            # next state 생성
            next_states[0, :, :,
                        idx] = self.tl_rl_memory[idx].next_state.detach().clone()
        # reward clear
        reward = self.reward.detach().clone()
        self.cum_reward += reward
        for idx, _ in enumerate(self.tl_rl_list):
            if idx not in torch.nonzero(mask_matrix).tolist():
                self.reward[0, idx] = torch.zeros_like(
                    self.reward[0, idx]).clone()
        return next_states  # list 반환 (안에 tensor)

    def step(self, action, mask_matrix, action_index_matrix, action_update_mask):
        '''
        매 초마다 action을 적용하고, next_state를 반환하는 역할
        yellow mask가 True이면 해당 agent reward저장
        '''
        # action update
        for index in torch.nonzero(mask_matrix):
            # action의 변환 -> 각 phase의 길이
            tl_rl = self.tl_rl_list[index]
            phase_length_set = self._toPhaseLength(
                tl_rl, action[0, index])
            # tls재설정
            tls = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                self.tl_rl_list[index])
            for phase_idx in self.traffic_node_info[tl_rl]['phase_index']:
                tls[0].phases[phase_idx].duration = phase_length_set[phase_idx]
            traci.trafficlight.setProgramLogic(tl_rl, tls[0])
            self.tl_rl_memory[index].action = action[0, index]
            # print(traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_rl_list[index])[0].phases)
            # print(phase_length_set)
        # action을 environment에 등록 후 상황 살피기,action을 저장
        # step
        traci.simulationStep()
        # for index in torch.nonzero(mask_matrix):
        #     tls=traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tl_rl_list[index])
        #     print(tls[0].phases,"after")

        self.before_action_update_mask = action_update_mask

    def calc_action(self, action_matrix, actions, mask_matrix):
        for index in torch.nonzero(mask_matrix):
            # print(self.traffic_node_info[self.tl_rl_list[0]
            #                                              ]['phase_duration'])
            # print(actions[0])
            phase_duration_list = self.traffic_node_info[self.tl_rl_list[index]
                                                         ]['phase_duration']
            pad_mat = torch.zeros_like(action_matrix[index])
            pad_mat_size = pad_mat.size()[1]

            new_phase_duration_list = self._toPhaseLength(
                self.tl_rl_list[index], actions[0, index])
            insert_mat = torch.tensor(
                new_phase_duration_list, dtype=torch.int, device=self.device)
            mat = torch.nn.functional.pad(
                insert_mat, (0, pad_mat_size-insert_mat.size()[0]), 'constant', 0)
            action_matrix[index] = mat

            # 누적 합산
            self.phase_action_matrix[index] = mat  # 누적합산이전 저장
            for l, _ in enumerate(phase_duration_list):
                if l >= 1:
                    action_matrix[index, l] += action_matrix[index, l-1]
            # print(action_matrix[0])
        return action_matrix.int()  # 누적합산 저장

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/reward', self.cum_reward.sum(),
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        # clear the value once in an epoch
        self.cum_reward = torch.zeros_like(self.cum_reward)

    def _toPhaseLength(self, tl_rl, action):  # action을 해석가능한 phase로 변환
        tl_dict = deepcopy(self.traffic_node_info[tl_rl])
        for j, idx in enumerate(tl_dict['phase_index']):
            tl_dict['phase_duration'][idx] = tl_dict['phase_duration'][idx] + \
                tl_dict['matrix_actions'][action[0, 0]][j] * \
                int((action[0, 1]+1)*1.5)
        phase_length_set = tl_dict['phase_duration']
        return phase_length_set
