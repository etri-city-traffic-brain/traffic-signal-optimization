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
    'gamma': 0.9,
    'tau': 0.001,
    'batch_size': 40,
    'experience_replay_size': 500000,
    'epsilon': 0.9,
    'epsilon_decay_rate': 0.99,
    'fc_net': [160, 160, 160, 160],
    'lr': 0.0001,
    'lr_decay_period': 50,
    'lr_decay_rate': 0.8,
    # 'lr_decay_rate': 0.995,
    'target_update_period': 10,
    'final_epsilon': 0.0012,
    'final_lr': 0.00001,
    'alpha': 0.91,
    'cnn':[50,60]
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class SuperQNetwork(nn.Module):
    def __init__(self, input_size, out_rate_size, out_time_size, configs):
        super(SuperQNetwork, self).__init__()
        self.configs = configs
        self.input_size = int(input_size)
        self.num_agent = len(self.configs['tl_rl_list'])
        self.state_space = self.configs['state_space']
        self.experience_replay = ReplayMemory(
            self.configs['experience_replay_size'])
        # Neural Net
        self.conv1 = nn.Conv2d(self.state_space, self.configs['cnn'][0], kernel_size=1)
        self.conv2 = nn.Conv2d(self.configs['cnn'][0], self.configs['cnn'][1], kernel_size=1)

        self.fc1 = nn.Linear(
            self.configs['cnn'][1]*4, self.configs['fc_net'][0])
        self.fc2 = nn.Linear(
            self.configs['fc_net'][0], self.configs['fc_net'][1])
        self.fc3 = nn.Linear(
            self.configs['fc_net'][1], self.configs['fc_net'][2])
        self.fc4 = nn.Linear(
            self.configs['fc_net'][2], self.configs['fc_net'][3])
        self.fc5 = nn.Linear(
            self.configs['fc_net'][3], out_rate_size)

        self.fc_y1 = nn.Linear(
            self.configs['cnn'][1]*4+1, self.configs['fc_net'][0])  # rate+state
        self.fc_y2 = nn.Linear(
            self.configs['fc_net'][0], self.configs['fc_net'][1])
        self.fc_y3 = nn.Linear(
            self.configs['fc_net'][1], self.configs['fc_net'][2])
        self.fc_y4 = nn.Linear(
            self.configs['fc_net'][2], self.configs['fc_net'][3])
        self.fc_y5 = nn.Linear(
            self.configs['fc_net'][3], out_time_size)

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.kaiming_uniform_(self.fc_y1.weight)
        nn.init.kaiming_uniform_(self.fc_y2.weight)
        nn.init.kaiming_uniform_(self.fc_y3.weight)
        nn.init.kaiming_uniform_(self.fc_y4.weight)

        if configs['mode'] == 'test':
            self.eval()

    def forward(self, input_x):
        # input_x = input_x.view(-1, self.state_space*4, 1)
        x_cnn = f.relu(self.conv1(input_x))
        x_cnn = f.relu(self.conv2(x_cnn))
        x_cnn = x_cnn.view(-1, self.configs['cnn'][1]*4)
        x_vehicle = f.relu(self.fc1(x_cnn))
        x_vehicle = f.relu(self.fc2(x_vehicle))
        x_vehicle = f.relu(self.fc3(x_vehicle))
        x_vehicle = f.relu(self.fc4(x_vehicle))
        rate_action_Q = self.fc5(x_vehicle)
        x_traffic = torch.cat((x_cnn, rate_action_Q.argmax(
            dim=1, keepdim=True).detach().clone()), dim=1).view(-1, self.configs['cnn'][1]*4+1)
        x_traffic = f.relu(self.fc_y1(x_traffic))
        x_traffic = f.relu(self.fc_y2(x_traffic))
        x_traffic = f.relu(self.fc_y3(x_traffic))
        x_traffic = f.relu(self.fc_y4(x_traffic))
        time_action_Q = self.fc_y5(x_traffic)
        return rate_action_Q, time_action_Q


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
        self.criterion = nn.MSELoss()
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.epsilon_decay_rate = self.configs['epsilon_decay_rate']
        self.batch_size = self.configs['batch_size']
        self.device = self.configs['device']
        self.running_loss = 0
        # self.writer=writer

        # NN composition
        # size에 따라 다르게 해주어야함
        self.rate_key_list = list()
        for i, key in enumerate(self.configs['traffic_node_info'].keys()):
            if configs['mode'] == 'train':
                rate_key = self.configs['traffic_node_info'][key]['num_phase']
            elif configs['mode'] == 'test':
                rate_key = str(
                    self.configs['traffic_node_info'][key]['num_phase'])
            self.rate_key_list.append(rate_key)

        self.mainSuperQNetwork = SuperQNetwork(
            self.state_space, self.rate_action_space[rate_key], self.time_action_space[0], self.configs)
        self.targetSuperQNetwork = SuperQNetwork(
            self.state_space, self.rate_action_space[rate_key], self.time_action_space[0], self.configs)
        # hard update, optimizer setting
        self.optimizer = optim.Adadelta(
            self.mainSuperQNetwork.parameters(), lr=self.configs['lr'])
        hard_update(self.targetSuperQNetwork, self.mainSuperQNetwork)
        self.lr_scheduler=optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=self.configs['lr_decay_period'],gamma=self.configs['lr_decay_rate'])

    def get_action(self, state, mask):
        # 전체를 날리는 epsilon greedy
        actions = torch.zeros((1, self.num_agent, self.action_size),
                              dtype=torch.int, device=self.device)
        with torch.no_grad():
            rate_actions = torch.zeros(
                (1, self.num_agent, 1), dtype=torch.int, device=self.device)
            time_actions = torch.zeros(
                (1, self.num_agent, 1), dtype=torch.int, device=self.device)
            for index in torch.nonzero(mask):
                if self.configs['mode'] == 'train':
                    if random.random() > self.epsilon:  # epsilon greedy
                        # masks = torch.cat((mask, mask), dim=0)
                        rate_action, time_action = self.mainSuperQNetwork(
                            state[0, :, :, index].view(-1, self.state_space, 4, 1))
                        rate_actions[0, index] = rate_action.max(1)[1].int()
                        time_actions[0, index] = time_action.max(1)[1].int()
                        # agent가 늘어나면 view(agents,action_size)
                    else:
                        rate_actions[0, index] = torch.tensor(random.randint(
                            0, self.rate_action_space[self.rate_key_list[index]]-1), dtype=torch.int, device=self.device)
                        time_actions[0, index] = torch.tensor(random.randint(
                            0, self.configs['time_action_space'][index]-1), dtype=torch.int, device=self.device)
                else:  # test
                    rate_action, time_action = self.mainSuperQNetwork(
                        state[0, :,:, index].view(-1, self.state_space,4, 1))
                    rate_actions[0, index] = rate_action.max(1)[1].int()
                    time_actions[0, index] = time_action.max(1)[1].int()

            actions = torch.cat((rate_actions, time_actions), dim=2)
        return actions

    def target_update(self,epoch):
        # Hard Update
        if epoch%self.configs['target_update_period']==0 and self.configs['update_type']=='hard':
            hard_update(self.targetSuperQNetwork, self.mainSuperQNetwork)

        # # Soft Update
        # Total Update
        if self.configs['update_type']=='soft':
            soft_update(self.targetSuperQNetwork,
                        self.mainSuperQNetwork, self.configs)

    def save_replay(self, state, action, reward, next_state, mask):
        for index in torch.nonzero(mask):
            # print(state[0,:, index])#,action[0,index],reward[0,index].sum(),next_state[0,:, index].sum())
            self.mainSuperQNetwork.experience_replay.push(
                state[0, :, :, index].view(-1, self.state_space, 4, 1), action[0, index], reward[0, index], next_state[0, :, :, index].view(-1, self.state_space, 4, 1))
            # print("state {}".format(state[0, :, :, index].view(-1, self.state_space, 4, 1)))
            # print("action {}".format(action[0, index]))
            # print("reward {}".format(reward[0, index]))
            # print("next_state {}".format(next_state[0, :, :, index].view(-1, self.state_space, 4, 1)))
            # if torch.eq(state[0, :, :, index],next_state[0, :, :, index]).sum()>0:
            #     print(torch.eq(state[0, :, :, index],next_state[0, :, :, index]).sum())
            #     print("FAKE")

    def update(self, mask):  # 각 agent마다 시행하기 # agent network로 돌아가서 시행 그러면될듯?
        # if mask.sum() > 0 and len(self.mainSuperQNetwork.experience_replay) > self.configs['batch_size']:
        if len(self.mainSuperQNetwork.experience_replay) > self.configs['batch_size'] and mask.sum() > 0:
            transitions = self.mainSuperQNetwork.experience_replay.sample(
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
            rate_state_action_values, time_state_action_values = self.mainSuperQNetwork(
                state_batch)
            rate_state_action_values = rate_state_action_values.gather(
                1, action_batch[:, 0].view(-1, 1).long())
            time_state_action_values = time_state_action_values.gather(
                1, action_batch[:, 1].view(-1, 1).long())
            # 모든 다음 상태를 위한 V(s_{t+1}) 계산
            rate_next_state_values = torch.zeros(
                self.configs['batch_size'], device=self.device, dtype=torch.float)
            time_next_state_values = torch.zeros(
                self.configs['batch_size'], device=self.device, dtype=torch.float)
            rate_Q, time_Q = self.mainSuperQNetwork(non_final_next_states)
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
            # total_loss= self.configs['alpha'] *rate_loss + (1.0-self.configs['alpha'])*time_loss
            self.running_loss += rate_loss/self.configs['batch_size']
            self.running_loss += time_loss/self.configs['batch_size']

            # 모델 최적화
            self.optimizer.zero_grad()
            # retain_graph를 하는 이유는 mainSuperQ에 대해 영향이 없게 하기 위함
            rate_loss.backward(retain_graph=True)
            time_loss.backward()
            # total_loss.backward(retain_graph=True)
            for param in self.mainSuperQNetwork.parameters():
                param.grad.data.clamp_(-1, 1)  # 값을 -1과 1로 한정시켜줌 (clipping)
            self.optimizer.step()

    def update_hyperparams(self, epoch):
        # decay rate (epsilon greedy)
        if self.epsilon > self.configs['final_epsilon']:
            self.epsilon *= self.epsilon_decay_rate

        # decay learning rate
        # if self.lr > self.configs['final_lr']:
        #     self.lr = self.lr_decay_rate*self.lr

        self.lr_scheduler.step()

    def save_weights(self, name):

        torch.save(self.mainSuperQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'Super.h5'))
        torch.save(self.targetSuperQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'Super_target.h5'))

    def load_weights(self, name):
        self.mainSuperQNetwork.load_state_dict(torch.load(os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'_{}Super.h5'.format(self.configs['replay_epoch']))))
        self.mainSuperQNetwork.eval()

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/loss', self.running_loss/self.configs['max_steps'],
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        writer.add_scalar('hyperparameter/lr', self.optimizer.param_groups[0]['lr'],
                          self.configs['max_steps']*epoch)
        writer.add_scalar('hyperparameter/epsilon',
                          self.epsilon, self.configs['max_steps']*epoch)
        # writer.add_histogram('action/time',self.time_action_dist_save, self.configs['max_steps']*epoch)
        # writer.add_histogram('action/rate',self.rate_action_dist_save, self.configs['max_steps']*epoch)

        # clear
        self.running_loss = 0