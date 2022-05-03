# -*- coding: utf-8 -*-

import numpy as np
import sys


import libsalt

from DebugConfiguration import DBG_OPTIONS

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

        # self.rewards = np.zeros(self.agent_num)  # env.rewards
        #
        # self.reward_related_info = list([] for i in range(self.agent_num))
        self.reset()

    def reset(self):
        '''
        reset reward related info
        :return:
        '''
        self.reward_related_info = list([] for i in range(self.agent_num))  # [ [], [], [],...]
        # self.rewards = np.zeros(self.agent_num)  # env.rewards
        self.rewards = list(0 for i in range(self.agent_num))
        print("new one")


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
        #      env/sappo_noConst.py : args.cp 관련 정리 필요
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




# todo hunsooni   self.reward_unit 에 따라 다르게 수집하게 하자.
class SaltRewardMgmtV2:
    '''
    class for reward management
     - gather info which will be used to calculate reward
     - calculate reward
    '''
    def __init__(self, reward_func, gather_unit, sa_obj, tl_obj, target_list, num_target=-1):
        '''
        constructor
        :param reward_func:
        :param gather_unit:
        :param sa_obj:
        :param target_list:
        :param num_target:
        '''

        self.DIC_CALC_REWARD, self.DIC_OP = self.__constructDictionaryForRewardCalculation()

        if reward_func not in self.DIC_CALC_REWARD.keys():
            raise Exception("internal error : should check dic_calc_reward")

        self.reward_func = reward_func
        self.reward_unit = gather_unit  # _REWARD_GATHER_UNIT_
        self.sa_obj = sa_obj
        self.tl_obj = tl_obj
        self.target_list = target_list  # SA name list
        self.num_target = len(target_list) if num_target == -1 else num_target
        # reward is only needed to train/test target SA

        self.agent_num = len(self.target_list)

        # convert <class 'numpy.ndarray'> to <class 'list'>
        # self.rewards = np.zeros(self.agent_num)  #  <class 'numpy.ndarray'>
        self.rewards = list(0 for i in range(self.agent_num)) # <class 'list'>

        # self.reward_related_info = list([] for i in range(self.agent_num))
        self.reward_related_info = []
        self.tl_reward_dic = {}  # reward for TL
        for sa_idx in range(self.num_target):
            sa_reward_info = []
            sa_name = self.target_list[sa_idx]
            tl_list = self.sa_obj[sa_name]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                sa_reward_info.append([])
                self.tl_reward_dic[tl_list[tl_idx]] = 0
            self.reward_related_info.append(sa_reward_info)
            # reward_related_info = [
            #                         [ [], [], [] ],     # for SA_a
            #                         [ [], [], [], [] ], #  for SA_b
            #                         [ [], [] ]          # for SA_c
            #                       ]




    def __constructDictionaryForRewardCalculation(self):
        '''
        construct dictionary for reward calculation
           key : reward function
           value : [func, weight] pairs to be used to calculate reward
        :return:
        '''
        dic_calc_reward = {}
        dic_calc_reward['cwq'] = ['sum', -1 / 5000]
        dic_calc_reward['pn'] = ['sum', 1]
        dic_calc_reward['tt'] = ['sum', -1]
        dic_calc_reward['wq'] = ['sum', -1 / 5000]
        dic_calc_reward['wq_max'] = ['max', -1]
        dic_calc_reward['wq_min'] = ['min', -1]
        dic_calc_reward['wt'] = ['sum', -1]
        dic_calc_reward['wt_ABV'] = ['mean', -1]
        dic_calc_reward['wt_median'] = ['median', -1]
        dic_calc_reward['wt_max'] = ['max', -1]
        dic_calc_reward['wt_SBV'] = ['sum', -1]
        dic_calc_reward['wt_SBV_max'] = ['max', -1]

        dic_op = {}
        dic_op['sum'] = np.sum
        dic_op['max'] = np.max
        dic_op['min'] = np.min
        dic_op['mean'] = np.mean

        return dic_calc_reward, dic_op



    def reset(self):
        for sa_idx in range(self.num_target):
            sa_name = self.target_list[sa_idx]
            tl_list = self.sa_obj[sa_name]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                self.reward_related_info[sa_idx][tl_idx].clear()
                self.tl_reward_dic[tl_list[tl_idx]] = 0
            self.rewards[sa_idx] = 0



    def gatherRewardRelatedInfo(self, action_t, simulation_steps, sim_period):
        # return self.gatherRewardRelatedInfoPerSA(action_t, simulation_steps, sim_period)
        if self.reward_unit == _REWARD_GATHER_UNIT_.SA:
            return self.gatherRewardRelatedInfoPerSA(action_t, simulation_steps, sim_period)
        else: # self.reward_unit == _REWARD_GATHER_UNIT_.TL
            return self.gatherRewardRelatedInfoPerTL(action_t, simulation_steps, sim_period)

    # todo delete comment : SA 단위로 보상 관련 정보 수집
    # def gatherRewardRelatedInfoPerSA_org(self, action_t, simulation_steps, sim_period):
    #     '''
    #     gather reward related info
    #     :param action_t: the unit time of green phase allowance (args.action_t)
    #                     how open do we check for signal change when action is KC(keep or change)
    #     :param simulation_steps:
    #     :param sim_period:
    #     :return:
    #     '''
    #     # from sappo_offset_single.py @ multi-agent/env
    #
    #     for sa_idx in range(self.num_target):
    #         sa = self.target_list[sa_idx]
    #
    #         link_list_0 = self.sa_obj[sa]['in_edge_list_0']
    #         link_list_1 = self.sa_obj[sa]['in_edge_list_1']
    #         lane_list_0 = self.sa_obj[sa]['in_lane_list_0']
    #         lane_list_1 = self.sa_obj[sa]['in_lane_list_1']
    #
    #         if self.reward_func == 'pn':
    #             for l in link_list_0:
    #                 self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
    #                                                              libsalt.link.getSumPassed(l))
    #             # for l in link_list_1:
    #             #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumPassed(l) * reward_weight)
    #         elif self.reward_func in ['wt', 'wt_max']:
    #             for l in link_list_0:
    #                 self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
    #                                                              libsalt.link.getAverageWaitingTime(l) / action_t)
    #             # for l in link_list_1:
    #             #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingTime(l) / self.actionT * reward_weight)
    #         elif self.reward_func in ['wq', 'wq_median', 'wq_min', 'wq_max', 'cwq']:
    #             for l in link_list_0:
    #                 # print("sum([l in x for x in lane_list_0])", sum([l in x for x in lane_list_0]))
    #                 self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
    #                                                              libsalt.link.getAverageWaitingQLength(l) * sum(
    #                                                                  [l in x for x in lane_list_0]))
    #             # for l in link_list_1:
    #             #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingQLength(l) * reward_weight)
    #         elif self.reward_func in ['wt_SBV', 'wt_SBV_max']:
    #             for l in link_list_0:
    #                 self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
    #                                                              libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l,
    #                                                                                                               simulation_steps) / 1000)
    #             # for l in link_list_1:
    #             #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps) / 1000 * reward_weight)
    #         elif self.reward_func == 'wt_ABV':
    #             for l in link_list_0:
    #                 self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
    #                                                              libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(
    #                                                                  l, simulation_steps) / 1000)
    #             # for l in link_list_1:
    #             #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulationSteps) / 1000 * reward_weight)
    #         elif self.reward_func == 'tt':
    #             for l in link_list_0:
    #                 # self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumTravelTime(l))
    #                 self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
    #                                                              libsalt.link.getSumTravelTime(l) / (
    #                                                                      len(link_list_0) * sim_period))
    #             # for l in link_list_1:
    #             #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumTravelTime(l) / 1000 * reward_weight)

    def gatherRewardRelatedInfoPerSA(self, action_t, simulation_steps, sim_period):
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

            # def __getRewardInfo(self, reward_info, link_list_0, lane_list_0, action_t, sim_steps, sim_period):

            self.reward_related_info[sa_idx] = self.__getRewardInfo(self.reward_related_info[sa_idx], link_list_0,
                                                                    lane_list_0, action_t, simulation_steps, sim_period)


    def __getRewardInfo(self, reward_info, link_list_0, lane_list_0, action_t, sim_steps, sim_period):

        if self.reward_func == 'pn':
            for l in link_list_0:
                reward_info = np.append(reward_info, libsalt.link.getSumPassed(l))

        elif self.reward_func in ['wt', 'wt_max']:
            for l in link_list_0:
                reward_info = np.append(reward_info, libsalt.link.getAverageWaitingTime(l) / action_t)
        elif self.reward_func in ['wq', 'wq_median', 'wq_min', 'wq_max', 'cwq']:
            for l in link_list_0:
                reward_info = np.append(reward_info,
                                        libsalt.link.getAverageWaitingQLength(l) * sum([l in x for x in lane_list_0]))
        elif self.reward_func in ['wt_SBV', 'wt_SBV_max']:
            for l in link_list_0:
                reward_info = np.append(reward_info,
                                        libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, sim_steps) / 1000)
        elif self.reward_func == 'wt_ABV':
            for l in link_list_0:
                reward_info = np.append(reward_info,
                                        libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, sim_steps) / 1000)
        elif self.reward_func == 'tt':
            for l in link_list_0:
                reward_info = np.append(reward_info,
                                        libsalt.link.getSumTravelTime(l) / (len(link_list_0) * sim_period))
        return list(reward_info)


    # todo delete comment : TL 단위로 보상 관련 정보 수집
    def gatherRewardRelatedInfoPerTL(self, action_t, simulation_steps, sim_period):
        '''
        gather reward related info per TL
        :param action_t: the unit time of green phase allowance (args.action_t)
                        how open do we check for signal change when action is KC(keep or change)
        :param simulation_steps:
        :param sim_period:
        :return:
        '''
        # from sappo_offset_single.py @ multi-agent/env

        for sa_idx in range(self.num_target):
            sa = self.target_list[sa_idx]

            tl_list = self.sa_obj[sa]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                link_list_0 = self.tl_obj[tl_list[tl_idx]]['in_edge_list_0']
                lane_list_0 = self.tl_obj[tl_list[tl_idx]]['in_lane_list_0']

                self.reward_related_info[sa_idx][tl_idx] = \
                    self.__getRewardInfo(self.reward_related_info[sa_idx][tl_idx], link_list_0,
                                                 lane_list_0, action_t, simulation_steps, sim_period)



    def __gatherTLRewardRelatedInfo(self, sa_idx, tl_idx, action_t, simulation_steps, sim_period):
        '''
        gather reward related info per TL
        :param action_t: the unit time of green phase allowance (args.action_t)
                        how open do we check for signal change when action is KC(keep or change)
        :param simulation_steps:
        :param sim_period:
        :return:
        '''
        # from sappo_offset_single.py @ multi-agent/env

        sa = self.target_list[sa_idx]

        tl_list = self.sa_obj[sa]['tlid_list']
        link_list_0 = self.tl_obj[tl_list[tl_idx]]['in_edge_list_0']
        lane_list_0 = self.tl_obj[tl_list[tl_idx]]['in_lane_list_0']

        tl_reward_info = []
        tl_reward_info = self.__getRewardInfo(tl_reward_info, link_list_0,
                                                 lane_list_0, action_t, simulation_steps, sim_period)

        return tl_reward_info


    def calculateReward(self, sa_idx):
        # return self.calculateRewardPerSA(sa_idx)
        if self.reward_unit == _REWARD_GATHER_UNIT_.SA:
            return self.calculateRewardPerSA(sa_idx)
        else: # self.reward_unit == _REWARD_GATHER_UNIT_.TL
            return self.calculateRewardPerTL_V2(sa_idx)

    # todo delete comment : SA 단위로 수집한 정보를 이용하여 보상 계산
    def calculateRewardPerSA(self, sa_idx):
        return self.calculateRewardPerSA_V2(sa_idx)

    def calculateRewardPerSA_V1(self, sa_idx):
        '''
        calculate reward
        :param sa_idx:
        :return:
        '''
        #
        # from sappo_offset_single.py @ multi-agent/env
        #
        #      env/sappo_noConst.py : args.cp 관련 정리 필요
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
            self.reward_related_info[sa_idx].clear()
            # todo hunsooni AttributeError: 'numpy.ndarray' object has no attribute 'clear'
            # self.reward_related_info[sa_idx] = []

    def calculateRewardPerSA_V2(self, sa_idx):
        op_str = self.DIC_CALC_REWARD[self.reward_func][0]
        op_func = self.DIC_OP[op_str]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]

        self.rewards[sa_idx] = op_func(self.reward_related_info[sa_idx])* weight

        # clear gathered reward related info
        # cumulate the reward related info when reward func is cwq
        if self.reward_func != 'cwq':
            self.reward_related_info[sa_idx].clear()


    # def __calculateRewardPerTLBySum(self, sa_idx, weight):
    #     '''
    #
    #     :param sa_idx:
    #     :param weight: +1, -1, 1/5000
    #     :return:
    #     '''
    #     sa_reward = 0  # for self.rewards[sa_idx]
    #     tl_reward = 0  # for self.tl_reward_dic[tlid]
    #     sa = self.target_list[sa_idx]
    #
    #     tl_list = self.sa_obj[sa]['tlid_list']
    #     num_tl = len(tl_list)
    #     for tl_idx in range(num_tl):
    #         tlid = tl_list[tl_idx]
    #         if len(self.reward_related_info[sa_idx][tl_idx]) == 0:
    #             tl_reward = 0
    #         else:
    #             tl_reward = np.sum(self.reward_related_info[sa_idx][tl_idx]) * weight
    #         self.tl_reward_dic[tlid] = tl_reward
    #         sa_reward += tl_reward
    #
    #     self.rewards[sa_idx] = sa_reward

    def __innerCalculateRewardPerTL(self, sa_idx, op_str, weight):
        self.__innerCalculateRewardPerTL_V2(sa_idx, op_str, weight)

    def __innerCalculateRewardPerTL_V1(self, sa_idx, op_str, weight):
        op_func = self.DIC_OP[op_str]
        sa_reward = 0  # for self.rewards[sa_idx]
        tl_reward = 0  # for self.tl_reward_dic[tlid]
        sa = self.target_list[sa_idx]

        tl_list = self.sa_obj[sa]['tlid_list']
        num_tl = len(tl_list)
        for tl_idx in range(num_tl):
            tlid = tl_list[tl_idx]
            if len(self.reward_related_info[sa_idx][tl_idx]) == 0:
                tl_reward = 0
            else:
                tl_reward = op_func(self.reward_related_info[sa_idx][tl_idx]) * weight
            self.tl_reward_dic[tlid] = tl_reward
            sa_reward += tl_reward

        self.rewards[sa_idx] = sa_reward


    def __innerCalculateRewardPerTL_V2(self, sa_idx, op_str, weight):
        op_func = self.DIC_OP[op_str]
        # sa_reward = 0  # for self.rewards[sa_idx]
        # tl_reward = 0  # for self.tl_reward_dic[tlid]
        sa = self.target_list[sa_idx]

        tl_list = self.sa_obj[sa]['tlid_list']
        num_tl = len(tl_list)
        sa_info = []
        for tl_idx in range(num_tl):
            tlid = tl_list[tl_idx]
            if len(self.reward_related_info[sa_idx][tl_idx]) == 0:
                tl_reward = 0
            else:
                tl_reward = op_func(self.reward_related_info[sa_idx][tl_idx]) * weight
            self.tl_reward_dic[tlid] = tl_reward
            sa_info += self.reward_related_info[sa_idx][tl_idx]

        sa_reward = op_func(sa_info) * weight

        self.rewards[sa_idx] = sa_reward


        if DBG_OPTIONS.PrintRewardMgmt:
            print(f'{sa} : {self.rewards[sa_idx]}')

            for tl_idx in range(num_tl):
                tlid = tl_list[tl_idx]
                print(f"\t{tlid} : {self.tl_reward_dic[tlid]}")



    # todo delete comment : TL 단위로 수집한 정보를 이용하여 보상 계산
    def calculateRewardPerTL(self, sa_idx):
        '''
        calculate reward
        :param sa_idx:
        :return:
        '''
        #
        # from sappo_offset_single.py @ multi-agent/env
        #
        #      env/sappo_noConst.py : args.cp 관련 정리 필요
        #
        self.rewards = 0
        self.tl_reward_dic = 0

        sa_reward = 0 # self.rewards[sa_idx]
        tl_reward = 0 # self.tl_reward_dic[sa_idx][tl_idx]
        dic_weight = {}
        dic_weight['sum'] = { 'cwq':-1/5000, 'pn' : 1, 'tt':-1, 'wt' : -1, 'wt_SBV' : -1, 'wq':-1/5000}
        dic_weight['max'] = {'wq_max':-1, 'wt_max':-1, 'wt_SBV_max':-1}
        dic_weight['min'] = {'wq_min':-1}
        dic_weight['mean'] = {'wt_ABV':-1}
        dic_weight['median'] = {'wt_median':-1}


        set_sum_as_reward = set(['cwq', 'pn', 'tt', 'wt', 'wt_SBV', 'wq'])
        set_max_as_reward = set(['wq_max', 'wt_max', 'wt_SBV_max'])
        set_min_as_reward = set(['wq_min'])
        set_mean_as_reward = set(['wt_ABV'])
        set_median_as_reward = set(['wt_median'])


        dic_calc_reward = {}
        dic_calc_reward['cwq'] = [np.sum, -2]

        dic_calc_reward['weight'] = {}

        if self.reward_func in set_sum_as_reward:
            self.__innerCalculateRewardPerTL(sa_idx, np.sum, dic_weight['sum'][self.reward_func])
        elif self.reward_func in set_max_as_reward:
            self.__innerCalculateRewardPerTL(sa_idx, np.max, dic_weight['max'][self.reward_func])
        elif self.reward_func in set_min_as_reward:
            self.__innerCalculateRewardPerTL(sa_idx, np.min, dic_weight['min'][self.reward_func])
        elif self.reward_func in set_mean_as_reward:
            self.__innerCalculateRewardPerTL(sa_idx, np.mean, dic_weight['mean'][self.reward_func])
        elif self.reward_func in set_median_as_reward:
            self.__innerCalculateRewardPerTL(sa_idx, np.median, dic_weight['median'][self.reward_func])

        # clear gathered reward related info
        # cumulate the reward related info when reward func is cwq
        if self.reward_func != 'cwq':
            sa_name = self.target_list[sa_idx]
            tl_list = self.sa_obj[sa_name]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                self.reward_related_info[sa_idx][tl_idx].clear()

    def calculateRewardPerTL_V2(self, sa_idx):
        '''
        calculate reward
        :param sa_idx:
        :return:
        '''
        #
        # from sappo_offset_single.py @ multi-agent/env
        #
        #      env/sappo_noConst.py : args.cp 관련 정리 필요
        #

        op = self.DIC_CALC_REWARD[self.reward_func][0]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]
        self.__innerCalculateRewardPerTL(sa_idx, op, weight)

        # clear gathered reward related info
        # cumulate the reward related info when reward func is cwq
        if self.reward_func != 'cwq':
            sa_name = self.target_list[sa_idx]
            tl_list = self.sa_obj[sa_name]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                self.reward_related_info[sa_idx][tl_idx].clear()



    # todo delete comment : 특정 교차로의 보상을 계산한다.
    def calculateTLReward(self, sa_idx, tlid, action_t, sim_step, sim_period):
        '''
        calculate rewards for given TL
        gather reward related info instantly

        :param sa_idx:
        :param tlid:
        :param action_t: the unit time of green phase allowance (args.action_t)
                        how open do we check for signal change when action is KC(keep or change)
        :param sim_step:
        :param sim_period:  SappoEnv.py 에있는 상수 이다. args 혹은 TRAIN_CONFIG에 넣어야 할 것 같다.
        :return:
        '''
        op_str = self.DIC_CALC_REWARD[self.reward_func][0]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]
        op_func = self.DIC_OP[op_str]
        sa = self.target_list[sa_idx]
        tl_list = self.sa_obj[sa]['tlid_list']

        tl_idx = tl_list.index(tlid)

        #todo hunsooni : do you think which one is proper?
        if 0:
            tl_reward_info = self.reward_related_info[sa_idx][tl_idx]
        else:
            tl_reward_info = self.__gatherTLRewardRelatedInfo(sa_idx, tl_idx, action_t, sim_step, sim_period)

        if len(tl_reward_info) == 0:
            tl_reward = 0
        else:
            tl_reward = op_func(tl_reward_info) * weight

        return tl_reward

