# -*- coding: utf-8 -*-

import numpy as np
import sys


import libsalt


from config import TRAIN_CONFIG
sys.path.append(TRAIN_CONFIG['libsalt_dir'])




class _REWARD_GATHER_UNIT_:
    '''
    reward gathering unit 보상을 어떤 단위로 수집할 것인가... 교차로, 교차로 그룹, ...
    '''
    TL = 0x00
    SA = 0x01
    ALL = 0x02




class SaltRewardMgmt :
    '''
    class for reward management
     - gather info which will be used to calculate reward
     - calculate reward
    '''
    def __init__(self, reward_func, gather_unit, sa_obj, target_list, num_target=-1):
        '''
        constructor
        :param reward_func:
        :param gather_unit:
        :param sa_obj:
        :param target_list:
        :param num_target:
        '''
        self.sa_obj = sa_obj
        self.reward_unit = gather_unit # _REWARD_GATHER_UNIT_
        self.reward_func = reward_func
        self.target_list = target_list
        self.num_target = len(target_list) if num_target == -1 else num_target
            # reward is only needed to train/test target SA

        self.agent_num = len(self.target_list)

        self.rewards = np.zeros(self.agent_num)  # env.rewards

        self.reward_related_info = list([] for i in range(self.agent_num))


    def reset(self):
        '''
        reset reward related info
        :return:
        '''
        self.reward_related_info = list([] for i in range(self.agent_num))  # [ [], [], [],...]
        self.rewards = np.zeros(self.agent_num)  # env.rewards



    def gatherRewardRelatedInfo(self, action_t, simulation_steps, sim_period):
        '''
        gather reward related info
        :param action_t: the unit time of green phase allowance (args.action_t)
                        how open do we check for signal change when action is KC(keep or change)
        :param simulation_steps:
        :param sim_period:
        :return:
        '''
        # from sappo_offset_single.py @ multi-agent/env

        for sa_idx in range(self.num_target):
            sa = self.target_list[sa_idx]

            link_list_0 = self.sa_obj[sa]['in_edge_list_0']
            link_list_1 = self.sa_obj[sa]['in_edge_list_1']
            lane_list_0 = self.sa_obj[sa]['in_lane_list_0']
            lane_list_1 = self.sa_obj[sa]['in_lane_list_1']

            if self.reward_func == 'pn':
                for l in link_list_0:
                    self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx], libsalt.link.getSumPassed(l))
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumPassed(l) * reward_weight)
            elif self.reward_func in ['wt', 'wt_max']:
                for l in link_list_0:
                    self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
                                                               libsalt.link.getAverageWaitingTime(l) / action_t)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingTime(l) / self.actionT * reward_weight)
            elif self.reward_func in ['wq', 'wq_median', 'wq_min', 'wq_max', 'cwq']:
                for l in link_list_0:
                    # print("sum([l in x for x in lane_list_0])", sum([l in x for x in lane_list_0]))
                    self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
                                                               libsalt.link.getAverageWaitingQLength(l) * sum(
                                                           [l in x for x in lane_list_0]))
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingQLength(l) * reward_weight)
            elif self.reward_func in ['wt_SBV', 'wt_SBV_max']:
                for l in link_list_0:
                    self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
                                                               libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l,
                                                                                                        simulation_steps) / 1000)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps) / 1000 * reward_weight)
            elif self.reward_func == 'wt_ABV':
                for l in link_list_0:
                    self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
                                                               libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(
                                                           l, simulation_steps) / 1000)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulationSteps) / 1000 * reward_weight)
            elif self.reward_func == 'tt':
                for l in link_list_0:
                    # self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumTravelTime(l))
                    self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
                                                               libsalt.link.getSumTravelTime(l) / (
                                                               len(link_list_0) * sim_period))
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumTravelTime(l) / 1000 * reward_weight)




    def calculateReward(self, sa_idx):
        '''
        calculate reward
        :param sa_idx:
        :return:
        '''
        #
        # from sappo_offset_single.py @ multi-agent/env
        #
        if self.reward_func == 'pn':
            self.rewards[sa_idx] = np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt':
            self.rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_max':
            self.rewards[sa_idx] = -np.max(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_SBV':
            self.rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_SBV_max':
            self.rewards[sa_idx] = -np.mean(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_ABV':
            self.rewards[sa_idx] = -np.mean(self.reward_related_info[sa_idx])

        elif self.reward_func == 'wq' or self.reward_func == 'cwq':
            self.rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx]) / 5000
        elif self.reward_func == 'wq_median':
            if len(self.reward_related_info[sa_idx]) == 0:
                self.rewards[sa_idx] = 0
            else:
                self.reward_related_info[sa_idx][self.reward_related_info[sa_idx] == 0] = np.nan
                if np.isnan(np.nanmedian(self.reward_related_info[sa_idx])):
                    self.rewards[sa_idx] = 0
                else:
                    self.rewards[sa_idx] = -np.nanmedian(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wq_min':
            self.rewards[sa_idx] = -np.min(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wq_max':
            if len(self.reward_related_info[sa_idx]) == 0:
                self.rewards[sa_idx] = 0
            else:
                self.rewards[sa_idx] = -np.max(self.reward_related_info[sa_idx])

        elif self.reward_func == 'tt':
            self.rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])

        # clear gathered reward related info
        # cumulate the reward related info when reward func is cwq
        if self.reward_func != 'cwq':
            # self.reward_related_info[sa_idx].clear()
            # todo hunsooni AttributeError: 'numpy.ndarray' object has no attribute 'clear'
            self.reward_related_info[sa_idx] = []


    def calculateRewardV2(self, sa_idx, args):
        #
        # from sappo_offset_single.py @ multi-agent/env
        #
        #      env/sappo_noConst.py : args.cp 관련 정리 필요
        if self.reward_func == 'pn':
            self.rewards[sa_idx] = np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt':
            self.rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_max':
            self.rewards[sa_idx] = -np.max(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_SBV':
            self.rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_SBV_max':
            self.rewards[sa_idx] = -np.mean(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_ABV':
            self.rewards[sa_idx] = -np.mean(self.reward_related_info[sa_idx])

        elif self.reward_func == 'wq' or self.reward_func == 'cwq':
            self.rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx]) / 5000
        elif self.reward_func == 'wq_median':
            if len(self.reward_related_info[sa_idx]) == 0:
                self.rewards[sa_idx] = 0
            else:
                self.reward_related_info[sa_idx][self.reward_related_info[sa_idx] == 0] = np.nan
                if np.isnan(np.nanmedian(self.reward_related_info[sa_idx])):
                    self.rewards[sa_idx] = 0
                else:
                    self.rewards[sa_idx] = -np.nanmedian(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wq_min':
            self.rewards[sa_idx] = -np.min(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wq_max':
            if len(self.reward_related_info[sa_idx]) == 0:
                self.rewards[sa_idx] = 0
            else:
                self.rewards[sa_idx] = -np.max(self.reward_related_info[sa_idx])

        elif self.reward_func == 'tt':
            self.rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])

        # clear gathered reward related info
        # cumulate the reward related info when reward func is cwq
        if self.reward_func != 'cwq':
            self.reward_related_info[sa_idx].clear()

        # ref. env/sappo_noConst.py
        # if args.action == "kc":
        #     self.rewards[sa_idx] -= np.sum(actions)*args.cp
