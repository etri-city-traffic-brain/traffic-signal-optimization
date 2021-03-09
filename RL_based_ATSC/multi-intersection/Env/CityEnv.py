import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy


class Memory():
    def __init__(self, configs):
        self.configs = configs
        self.reward = torch.zeros(1, dtype=torch.float, device=configs['device'])
        self.state = torch.zeros(
            (1, self.configs['state_space'], len(configs['tl_rl_list'])), dtype=torch.float, device=configs['device'])
        self.next_state = torch.zeros_like(self.state)
        self.action = torch.zeros(
            (1, self.configs['action_size']), dtype=torch.int, device=configs['device'])


class CityEnv(baseEnv):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.device=self.configs['device']
        self.phase_action_matrix = torch.zeros(  # 누적합산 이전의 action_matrix
            (self.configs['num_agent'], self.configs['max_phase_num']), dtype=torch.int, device=self.device)  # reward 계산시 사용
        self.tl_list = traci.trafficlight.getIDList()
        self.tl_rl_list = self.configs['tl_rl_list']
        self.num_agent = len(self.tl_rl_list)
        self.side_list = ['u', 'r', 'd', 'l']
        self.interest_list = self.configs['interest_list']
        self.node_interest_pair = self.configs['node_interest_pair']

        self.reward = 0
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

        # traci 내의 정보받아오기
        for key in self.traffic_node_info.keys():
            self.traffic_node_info[key]['program'] = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                key)

        self.left_lane_num_dict = dict()
        # lane 정보 저장
        for interest in self.node_interest_pair:
            # 모든 inflow에 대해서
            for pair in self.node_interest_pair[interest]:
                if pair['inflow'] == None:
                    continue
                self.left_lane_num_dict[pair['inflow']] = traci.edge.getLaneNumber(
                    pair['inflow'])-1

    def get_state(self, mask):
        '''
        매 주기마다 매 주기 이전의 state, 현재 state, reward를 반환하는 함수
        reward,next_state<-state 초기화 해줘야됨
        '''

        state = torch.zeros(
            (1, self.num_agent, self.state_space, self.num_agent), dtype=torch.float, device=self.device)
        next_state = torch.zeros_like(state)
        action = torch.zeros(
            (1, self.num_agent, self.num_agent, 2), dtype=torch.int, device=self.device)
        reward = torch.zeros((1, self.num_agent),
                             dtype=torch.float, device=self.device)
        for index in torch.nonzero(mask):
            state[0, index, :] = deepcopy(self.tl_rl_memory[index].state)
            action[0, index, :] = deepcopy(self.tl_rl_memory[index].action)
            next_state[0, index] = deepcopy(
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

        # Reward
        for index in torch.nonzero(action_change_mask):
            outflow = 0
            inflow = 0
            interests = self.node_interest_pair[self.tl_rl_list[index]]
            for interest in interests:
                if interest['outflow']:  # None이 아닐 때 행동
                    outflow += (traci.edge.getLastStepVehicleNumber(
                        interest['outflow'])-traci.edge.getLastStepHaltingNumber(
                        interest['outflow']))/100.0
                if interest['inflow']:  # None이 아닐 때 행동
                    inflow += traci.edge.getLastStepHaltingNumber(
                        interest['inflow'])/100.0
            # pressure=inflow-outflow
            # reward cumulative sum
            pressure = torch.tensor(
                (inflow-outflow), dtype=torch.float, device=self.device)/100.0
            self.tl_rl_memory[index].reward -= pressure
            self.reward -= pressure
        # # penalty
        # if mask_matrix.sum() > 0:  # phase 어긋날 때 문제
        #     # save reward
        #     for index in torch.nonzero(mask_matrix):
        #         if self.phase_action_matrix[index].sum() != 0:
        #             phase_index = torch.tensor(
        #                 self.configs['traffic_node_info'][self.tl_rl_list[index]]['phase_index'], device=self.device).view(1, -1).long()
        #             # penalty for phase duration more than maxDuration
        #             if torch.gt(self.phase_action_matrix[index].gather(dim=1, index=phase_index), torch.tensor(self.configs['traffic_node_info'][self.tl_rl_list[index]]['max_phase'])).sum():
        #                 self.tl_rl_memory[index].reward -= 1  # penalty
        #                 self.reward -= 1
        #             # penalty for phase duration less than minDuration
        #             if torch.gt(torch.tensor(self.configs['traffic_node_info'][self.tl_rl_list[index]]['min_phase']), self.phase_action_matrix[index].gather(dim=1, index=phase_index)).sum():
        #                 self.tl_rl_memory[index].reward -= 1  # penalty
        #                 self.reward -= 1

        # action 변화를 위한 state
        if mask_matrix.sum() > 0:  # 검색의 필요가 없다면 검색x
            next_state = list()
            # 모든 rl node에 대해서
            # vehicle state
            for interest in self.node_interest_pair:
                veh_state = torch.zeros(
                    (self.state_space, 1), dtype=torch.float, device=self.device)
                # 모든 inflow에 대해서
                for j, pair in enumerate(self.node_interest_pair[interest]):
                    if pair['inflow'] is None:
                        veh_state[j*2] = 0.0
                        veh_state[j*2+1] = 0.0
                    else:
                        left_movement = traci.lane.getLastStepHaltingNumber(
                            pair['inflow']+'_{}'.format(self.left_lane_num_dict[pair['inflow']]))/100.0  # 멈춘애들 계산
                        # 직진
                        veh_state[j*2] = traci.edge.getLastStepHaltingNumber(
                            pair['inflow'])/100.0-left_movement  # 가장 좌측에 멈춘 친구를 왼쪽차선 이용자로 판단
                        # 좌회전
                        veh_state[j*2+1] = left_movement
                next_state.append(veh_state)
            next_states = torch.cat(next_state, dim=1).view(
                1, self.state_space, self.num_agent)
            # 각 agent env에 state,next_state 저장
            for state_index in torch.nonzero(mask_matrix):
                self.tl_rl_memory[state_index].state = self.tl_rl_memory[state_index].next_state
                self.tl_rl_memory[state_index].next_state = next_states
        else:
            next_states = torch.zeros(
                1, self.state_space, self.num_agent, dtype=torch.float, device=self.device)
        return next_states

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
            tls = self.traffic_node_info[tl_rl]['program']
            for phase_idx in self.traffic_node_info[tl_rl]['phase_index']:
                tls[0].phases[phase_idx].duration = phase_length_set[phase_idx]
            traci.trafficlight.setProgramLogic(tl_rl, tls[0])
            self.tl_rl_memory[index].action = action.int()
        # action을 environment에 등록 후 상황 살피기,action을 저장

        # step
        traci.simulationStep()
        # next state 받아오기, reward저장
        next_state = self.collect_state(
            action_update_mask, action_index_matrix, mask_matrix)
        self.before_action_update_mask = action_update_mask
        return next_state

    def calc_action(self, action_matrix, actions, mask_matrix):
        for index in torch.nonzero(mask_matrix):
            actions = actions.long()
            phase_duration_list = self.traffic_node_info[self.tl_rl_list[index]
                                                         ]['phase_duration']
            pad_mat = torch.zeros_like(action_matrix[index])
            pad_mat_size = pad_mat.size()[1]
            insert_mat = torch.tensor(
                phase_duration_list, dtype=torch.int, device=self.device)
            mat = torch.nn.functional.pad(
                insert_mat, (0, pad_mat_size-insert_mat.size()[0]), 'constant', 0)
            action_matrix[index] = mat
            # action_matrix[index] = torch.tensor(
            #     phase_duration_list, dtype=torch.int, device=self.device)
            # 누적 합산
            self.phase_action_matrix[index] = mat  # 누적합산이전 저장
            for l, _ in enumerate(phase_duration_list):
                if l >= 1:
                    action_matrix[index, l] += action_matrix[index, l-1]
        return action_matrix.int()  # 누적합산 저장

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/reward', self.reward,
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        # clear the value once in an epoch
        self.reward = 0

    def _toPhaseLength(self, tl_rl, action):  # action을 해석가능한 phase로 변환
        tl_dict = deepcopy(self.traffic_node_info[tl_rl])
        for j, idx in enumerate(tl_dict['phase_index']):
            tl_dict['phase_duration'][idx] = tl_dict['phase_duration'][idx] + \
                tl_dict['matrix_actions'][action[0, 0]][j] * action[0, 1]
        phase_length_set = tl_dict['phase_duration']
        return phase_length_set
