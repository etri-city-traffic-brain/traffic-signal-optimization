import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
import random
import os
from collections import namedtuple
from copy import deepcopy
from Agent.base import RLAlgorithm, ReplayMemory, merge_dict, hard_update, merge_dict_non_conflict, soft_update
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
DEFAULT_CONFIG = {
    'gamma': 0.99,
    'tau': 0.001,
    'batch_size': 8,
    'experience_replay_size': 1e5,
    'epsilon': 0.8,
    'epsilon_decay_rate': 0.99,
    'fc_net': [36, 48, 24],
    'lr': 1e-3,
    'lr_decay_rate': 0.99,
    'target_update_period': 20,
    'final_epsilon': 0.0005,
    'final_lr': 1e-7,
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class QNetwork(nn.Module):
    def __init__(self, input_size, rate_output_size, time_output_size, configs):
        super(QNetwork, self).__init__()
        self.configs = configs
        self.input_size = int(input_size)
        self.rate_output_size = int(rate_output_size)
        self.time_output_size = int(time_output_size)
        self.num_agent = len(configs['tl_rl_list'])

        # build nn 증감
        self.fc1 = nn.Linear(self.input_size, self.configs['fc_net'][0])
        self.fc2 = nn.Linear(
            self.configs['fc_net'][0], self.configs['fc_net'][1])
        self.fc3 = nn.Linear(
            self.configs['fc_net'][1], self.configs['fc_net'][2])
        self.fc4 = nn.Linear(self.configs['fc_net'][2], self.rate_output_size)
        # 증감의 크기
        # +1은 증감에서의 argmax value
        self.fc_y1 = nn.Linear(self.input_size+1, self.configs['fc_net'][0])
        self.fc_y2 = nn.Linear(
            self.configs['fc_net'][0], self.configs['fc_net'][1])
        self.fc_y3 = nn.Linear(
            self.configs['fc_net'][1], self.configs['fc_net'][2])
        self.fc_y4 = nn.Linear(
            self.configs['fc_net'][2], self.time_output_size)

        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.xavier_uniform(self.fc4.weight)
        nn.init.xavier_uniform(self.fc_y1.weight)
        nn.init.xavier_uniform(self.fc_y2.weight)
        nn.init.xavier_uniform(self.fc_y3.weight)
        nn.init.xavier_uniform(self.fc_y4.weight)
        if configs['mode'] == 'test':
            self.eval()

        # Experience Replay
        self.experience_replay = ReplayMemory(
            self.configs['experience_replay_size'])

    def forward(self, input_x):
        # 증감
        x = f.relu(self.fc1(input_x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        # 증감의 크기
        # argmax(x)값을 구해서 넣기
        y = torch.cat((input_x, x.max(dim=1)[1].view(-1, 1).detach()), dim=1)
        y = f.relu(self.fc_y1(y))
        y = f.relu(self.fc_y2(y))
        y = f.relu(self.fc_y3(y))
        y = self.fc_y4(y)
        return x, y  # q value

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.push(
            state, action, reward, next_state)  # asynchronous하게 저장하고 불러오기


class SuperQNetwork(nn.Module):
    def __init__(self, input_size, output_size, configs):
        super(SuperQNetwork, self).__init__()
        self.configs = configs
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_agent = len(self.configs['tl_rl_list'])
        self.state_space = self.configs['state_space']
        # Neural Net
        self.conv1 = nn.Conv1d(self.state_space, 4, kernel_size=1)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=1)
        self.fc1 = nn.Linear(
            self.input_size*4, int(self.state_space*1.5*self.num_agent))
        self.fc2 = nn.Linear(
            int(self.state_space*1.5*self.num_agent), int(self.state_space*1.5*self.num_agent))
        self.fc3 = nn.Linear(
            int(self.state_space*1.5*self.num_agent), int(self.state_space*1*self.num_agent))
        self.fc4 = nn.Linear(
            self.state_space*1*self.num_agent, self.output_size)
        
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.xavier_uniform(self.fc4.weight)

        if configs['mode'] == 'test':
            self.eval()

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = x.view(-1, self.num_agent*4)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, 0.4)
        x = f.relu(self.fc2(x))
        x = f.dropout(x, 0.3)
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))  # .view(-1, self.num_agent,
        #                      int(self.configs['state_space']/2))
        return x.view(-1, self.output_size)


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        super().__init__(configs)
        if configs['mode'] == 'train' or configs['mode'] == 'simulate':
            os.mkdir(os.path.join(
                self.configs['current_path'], 'training_data', self.configs['time_data'], 'model'))
            self.configs = merge_dict(configs, DEFAULT_CONFIG)
        else:  # test
            self.configs = merge_dict_non_conflict(configs, DEFAULT_CONFIG)
        self.num_agent = len(self.configs['tl_rl_list'])
        self.state_space = self.configs['state_space']

        # action space
        # rate action space
        self.rate_action_space = self.configs['rate_action_space']
        # time action space
        self.time_action_space = self.configs['time_action_space']
        self.action_size = self.configs['action_size']
        self.gamma = self.configs['gamma']
        self.epsilon = self.configs['epsilon']
        self.criterion = nn.SmoothL1Loss()
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.epsilon_decay_rate = self.configs['epsilon_decay_rate']
        self.batch_size = self.configs['batch_size']
        self.device=self.configs['device']
        self.running_loss = 0
        self.super_output_size = int(self.num_agent*2)
        self.super_input_size = int(self.num_agent)
        # NN composition
        self.mainSuperQNetwork = SuperQNetwork(
            self.super_input_size, self.super_output_size, self.configs)
        self.targetSuperQNetwork = SuperQNetwork(
            self.super_input_size, self.super_output_size, self.configs)
        # size에 따라 다르게 해주어야함
        self.mainQNetwork = list()
        self.targetQNetwork = list()
        self.rate_key_list = list()
        for i, key in enumerate(self.configs['traffic_node_info'].keys()):
            if configs['mode'] == 'train':
                rate_key = self.configs['traffic_node_info'][key]['num_phase']
            elif configs['mode'] == 'test':
                rate_key = str(
                    self.configs['traffic_node_info'][key]['num_phase'])
            self.rate_key_list.append(rate_key)
            self.mainQNetwork.append(QNetwork(
                self.super_output_size, self.rate_action_space[rate_key], self.time_action_space[i], self.configs))
            self.targetQNetwork.append(QNetwork(
                self.super_output_size, self.rate_action_space[rate_key], self.time_action_space[i], self.configs))

        # hard update, optimizer setting
        self.optimizer = list()
        hard_update(self.targetSuperQNetwork, self.mainSuperQNetwork)
        for targetQ, mainQ in zip(self.targetQNetwork, self.mainQNetwork):
            hard_update(targetQ, mainQ)
            params = chain(self.mainSuperQNetwork.parameters(),
                           mainQ.parameters())
            self.optimizer.append(optim.Adam(params, lr=self.lr))

        # Network
        print("========SUPER NETWORK==========\n", self.mainSuperQNetwork)
        print("========NETWORK==========\n")
        for i in range(self.num_agent):
            print(self.mainQNetwork[i])

    def get_action(self, state, mask):

        # 전체를 날리는 epsilon greedy
        actions = torch.zeros((1, self.num_agent, self.action_size),
                              dtype=torch.int, device=self.device)
        with torch.no_grad():
            if mask.sum() > 0:
                obs = self.mainSuperQNetwork(state)
                rate_actions = torch.zeros(
                    (1, self.num_agent, 1), dtype=torch.int, device=self.device)
                time_actions = torch.zeros(
                    (1, self.num_agent, 1), dtype=torch.int, device=self.device)
                for index in torch.nonzero(mask):
                    if random.random() > self.epsilon:  # epsilon greedy
                        # masks = torch.cat((mask, mask), dim=0)
                        rate_action, time_action = self.mainQNetwork[index](
                            obs)
                        rate_actions[0, index] = rate_action.max(1)[1].int()
                        time_actions[0, index] = time_action.max(1)[1].int()
                        # agent가 늘어나면 view(agents,action_size)
                    else:
                        rate_actions[0, index] = torch.tensor(random.randint(
                            0, self.rate_action_space[self.rate_key_list[index]]-1), dtype=torch.int, device=self.device)
                        time_actions[0, index] = torch.tensor(random.randint(
                            0, self.configs['time_action_space'][index]-1), dtype=torch.int, device=self.device)
                actions = torch.cat((rate_actions, time_actions), dim=2)
        return actions

    def target_update(self):
        # Hard Update
        for target, source in zip(self.targetQNetwork, self.mainQNetwork):
            hard_update(target, source)
        # Total Update
        hard_update(self.targetSuperQNetwork, self.mainSuperQNetwork)

        # # Soft Update
        # for target, source in zip(self.targetQNetwork, self.mainQNetwork):
        #     soft_update(target, source, self.configs)
        # # Total Update
        # soft_update(self.targetSuperQNetwork,
        #             self.mainSuperQNetwork, self.configs)

    def save_replay(self, state, action, reward, next_state, mask):
        for i in torch.nonzero(mask):
            self.mainQNetwork[i].experience_replay.push(
                state[0, i], action[0, i], reward[0, i], next_state[0, i])

    def update(self, mask):  # 각 agent마다 시행하기 # agent network로 돌아가서 시행 그러면될듯?
        for i, (mainQNetwork, targetQNetwork, optimizer) in enumerate(zip(self.mainQNetwork, self.targetQNetwork, self.optimizer)):
            if mask[i] == False or len(mainQNetwork.experience_replay) < self.configs['batch_size']:
                continue
            transitions = mainQNetwork.experience_replay.sample(
                self.configs['batch_size'])
            batch = Transition(*zip(*transitions))

            # 최종 상태가 아닌 마스크를 계산하고 배치 요소를 연결합니다.
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)

            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None], dim=0)

            # dim=0인 이유는 batch 끼리 cat 하는 것이기 때문임
            state_batch = torch.cat(batch.state)

            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            # print(state_batch[0],action_batch[0],reward_batch[0],non_final_mask[0])

            # Q(s_t, a) 계산 - 모델이 action batch의 a'일때의 Q(s_t,a')를 계산할때, 취한 행동 a'의 column 선택(column이 Q)
            rate_state_action_values, time_state_action_values = mainQNetwork(self.mainSuperQNetwork(
                state_batch))
            rate_state_action_values = rate_state_action_values.gather(
                1, action_batch[:, 0, 0].view(-1, 1).long())
            time_state_action_values = time_state_action_values.gather(
                1, action_batch[:, 0, 1].view(-1, 1).long())
            # 모든 다음 상태를 위한 V(s_{t+1}) 계산
            rate_next_state_values = torch.zeros(
                self.configs['batch_size'], device=self.device, dtype=torch.float)
            time_next_state_values = torch.zeros(
                self.configs['batch_size'], device=self.device, dtype=torch.float)
            rate_Q, time_Q = targetQNetwork(
                self.mainSuperQNetwork(non_final_next_states))
            rate_next_state_values[non_final_mask] = rate_Q.max(
                1)[0].detach().to(self.device)
            time_next_state_values[non_final_mask] = time_Q.max(1)[0].detach().to(
                self.device)  # .to(self.configs['device'])  # 자신의 Q value 중에서max인 value를 불러옴

            # 기대 Q 값 계산
            rate_expected_state_action_values = (
                rate_next_state_values * self.configs['gamma']) + reward_batch
            time_expected_state_action_values = (
                time_next_state_values * self.configs['gamma']) + reward_batch

            # loss 계산
            rate_loss = self.criterion(rate_state_action_values,
                                       rate_expected_state_action_values.unsqueeze(1))
            time_loss = self.criterion(time_state_action_values,
                                       time_expected_state_action_values.unsqueeze(1))
            self.running_loss += rate_loss/self.configs['batch_size']
            self.running_loss += time_loss/self.configs['batch_size']

            # 모델 최적화
            optimizer.zero_grad()
            # retain_graph를 하는 이유는 mainSuperQ에 대해 영향이 없게 하기 위함
            rate_loss.backward(retain_graph=True)
            time_loss.backward()
            for param in mainQNetwork.parameters():
                param.grad.data.clamp_(-1, 1)  # 값을 -1과 1로 한정시켜줌 (clipping)
            optimizer.step()

    def update_hyperparams(self, epoch):
        # decay rate (epsilon greedy)
        if self.epsilon > self.configs['final_epsilon']:
            self.epsilon *= self.epsilon_decay_rate

        # decay learning rate
        if self.lr > self.configs['final_lr']:
            self.lr = self.lr_decay_rate*self.lr

    def save_weights(self, name):

        torch.save(self.mainSuperQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'Super.h5'))
        torch.save(self.targetSuperQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'Super_target.h5'))

        for i, (mainQ, targetQ) in enumerate(zip(self.mainQNetwork, self.targetQNetwork)):
            torch.save(mainQ.state_dict(), os.path.join(
                self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'_{}.h5'.format(i)))
            torch.save(targetQ.state_dict(), os.path.join(
                self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'_target_{}.h5'.format(i)))

    def load_weights(self, name):
        self.mainSuperQNetwork.load_state_dict(torch.load(os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'_{}Super.h5'.format(self.configs['replay_epoch']))))
        self.mainSuperQNetwork.eval()
        for i, mainQ in enumerate(self.mainQNetwork):
            mainQ.load_state_dict(torch.load(os.path.join(
                self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'_{}_{}.h5'.format(self.configs['replay_epoch'], i))))
            mainQ.eval()

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/loss', self.running_loss/self.configs['max_steps'],
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        writer.add_scalar('hyperparameter/lr', self.lr,
                          self.configs['max_steps']*epoch)
        writer.add_scalar('hyperparameter/epsilon',
                          self.epsilon, self.configs['max_steps']*epoch)

        # clear
        self.running_loss = 0
