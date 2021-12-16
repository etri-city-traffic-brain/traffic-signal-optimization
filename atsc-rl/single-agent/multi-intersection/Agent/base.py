import os
import json
import torch
from torch import nn
import copy
import random
from collections import namedtuple
from copy import deepcopy


class RLAlgorithm():
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def get_action(self, state):
        '''
        return action (torch Tensor (1,action_space))
        상속을 위한 함수
        '''
        raise NotImplementedError

    def update_hyperparams(self, epoch):
        '''
        상속을 위한 함수
        '''
        raise NotImplementedError

    def update_tensorboard(self, writer, epoch):
        '''
        상속을 위한 함수
        '''
        raise NotImplementedError

    def save_params(self, time_data):
        with open(os.path.join(self.configs['current_path'], 'training_data', '{}.json'.format(time_data)), 'w') as fp:
            json.dump(self.configs, fp, indent=2)

    def load_params(self, file_name):
        ''' replay_name from flags.replay_name '''
        with open(os.path.join(self.configs['current_path'], 'training_data', '{}.json'.format(file_name)), 'r') as fp:
            configs = json.load(fp)
        return configs


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """전환 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def merge_dict(d1, d2):
    '''
    d2를 d1위에 덮기 (충돌 ver)
    '''
    merged = copy.deepcopy(d1)
    for key in d2.keys():
        if key in merged.keys():
            print(key)
            raise KeyError
        merged[key] = d2[key]
    return merged


def merge_dict_non_conflict(d1, d2):
    '''
    d2를 d1위에 덮기 (non 충돌 ver)
    '''
    merged = copy.deepcopy(d1)
    for key in d2.keys():
        merged[key] = d2[key]
    return merged


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target,source,configs):
    for target_param,param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*(1.0-configs['tau'])+param.data*configs['tau'])