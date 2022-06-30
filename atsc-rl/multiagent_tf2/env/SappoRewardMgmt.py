# -*- coding: utf-8 -*-

import numpy as np
from deprecated import deprecated

import libsalt

from DebugConfiguration import DBG_OPTIONS
from TSOConstants import _REWARD_GATHER_UNIT_


@deprecated(reason="use another Class : SaltRewardMgmtV3")
class SaltRewardMgmtV1 :
    '''
    class for reward management
     - gather info which will be used to calculate reward
     - calculate reward
    '''
    def __init__(self, reward_func, gather_unit, sa_obj, target_name_list, num_target=-1):
        '''
        constructor
        :param reward_func:
        :param gather_unit:
        :param sa_obj:
        :param target_name_list:
        :param num_target:
        '''
        self.sa_obj = sa_obj
        self.reward_unit = gather_unit.upper() # _REWARD_GATHER_UNIT_
        self.reward_func = reward_func
        self.target_name_list = target_name_list
        self.num_target = len(target_name_list) if num_target == -1 else num_target
            # reward is only needed to train/test target SA

        self.agent_num = len(self.target_name_list)

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
            sa = self.target_name_list[sa_idx]

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
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulation_steps) / 1000 * reward_weight)
            elif self.reward_func == 'wt_ABV':
                for l in link_list_0:
                    self.reward_related_info[sa_idx] = np.append(self.reward_related_info[sa_idx],
                                                               libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(
                                                           l, simulation_steps) / 1000)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulation_steps) / 1000 * reward_weight)
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
            # todo AttributeError: 'numpy.ndarray' object has no attribute 'clear'
            self.reward_related_info[sa_idx] = []




# self.reward_unit 에 따라 다르게 수집하게 하자.
@deprecated(reason="use another Class : SaltRewardMgmtV3")
class SaltRewardMgmtV2:
    '''
    class for reward management
     - gather info which will be used to calculate reward
     - calculate reward
    '''
    def __init__(self, reward_func, gather_unit, action_t, reward_info_collection_cycle, sa_obj, tl_obj, target_name_list, num_target=-1):
        '''
        constructor
        :param reward_func:
        :param gather_unit:
        :param sa_obj:
        :param target_name_list:
        :param num_target:
        '''

        self.DIC_CALC_REWARD, self.DIC_OP_FUNC = self.__constructDictionaryForRewardCalculation()

        if reward_func not in self.DIC_CALC_REWARD.keys():
            raise Exception("internal error : should check dic_calc_reward")

        self.reward_func = reward_func
        self.reward_unit = gather_unit.upper()  # _REWARD_GATHER_UNIT_
        self.action_t = action_t
        self.reward_info_collection_cycle = reward_info_collection_cycle
        self.sa_obj = sa_obj
        self.tl_obj = tl_obj
        self.target_name_list = target_name_list  # SA name list
        self.num_target = len(target_name_list) if num_target == -1 else num_target
        # reward is only needed to train/test target SA

        self.agent_num = len(self.target_name_list)

        # convert <class 'numpy.ndarray'> to <class 'list'>
        # self.rewards = np.zeros(self.agent_num)  #  <class 'numpy.ndarray'>
        self.sa_rewards = list(0 for i in range(self.agent_num)) # <class 'list'>

        # self.reward_related_info = list([] for i in range(self.agent_num))
        self.reward_related_info = []
        self.tl_reward_dic = {}  # reward for TL
        for sa_idx in range(self.num_target):
            sa_reward_info = []
            sa_name = self.target_name_list[sa_idx]
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

        dic_op_func = {}
        dic_op_func['sum'] = np.sum
        dic_op_func['max'] = np.max
        dic_op_func['min'] = np.min
        dic_op_func['mean'] = np.mean

        return dic_calc_reward, dic_op_func



    def reset(self):
        for sa_idx in range(self.num_target):
            sa_name = self.target_name_list[sa_idx]
            tl_list = self.sa_obj[sa_name]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                self.reward_related_info[sa_idx][tl_idx].clear()
                self.tl_reward_dic[tl_list[tl_idx]] = 0
            self.sa_rewards[sa_idx] = 0



    def gatherRewardRelatedInfo(self, simulation_steps):
        # return self.gatherRewardRelatedInfoPerSA(action_t, simulation_steps, reward_info_collection_cycle)
        if self.reward_unit == _REWARD_GATHER_UNIT_.SA:
            return self.gatherRewardRelatedInfoPerSA(simulation_steps)
        else: # self.reward_unit == _REWARD_GATHER_UNIT_.TL
            return self.gatherRewardRelatedInfoPerTL(simulation_steps)



    def gatherRewardRelatedInfoPerSA(self, simulation_steps):
        '''
        gather reward related info
        :param action_t: the unit time of green phase allowance (args.action_t)
                        how open do we check for signal change when action is KC(keep or change)
        :param simulation_steps:
        :param reward_info_collection_cycle:
        :return:
        '''
        # from sappo_offset_single.py @ multi-agent/env

        for sa_idx in range(self.num_target):
            sa = self.target_name_list[sa_idx]

            link_list_0 = self.sa_obj[sa]['in_edge_list_0']
            link_list_1 = self.sa_obj[sa]['in_edge_list_1']
            lane_list_0 = self.sa_obj[sa]['in_lane_list_0']
            lane_list_1 = self.sa_obj[sa]['in_lane_list_1']

            self.reward_related_info[sa_idx] = self.__getRewardInfo(self.reward_related_info[sa_idx], link_list_0,
                                                                    lane_list_0, simulation_steps)


    def __getRewardInfo(self, reward_info, link_list_0, lane_list_0, sim_steps):

        if self.reward_func == 'pn':
            for l in link_list_0:
                reward_info = np.append(reward_info, libsalt.link.getSumPassed(l))

        elif self.reward_func in ['wt', 'wt_max']:
            for l in link_list_0:
                reward_info = np.append(reward_info, libsalt.link.getAverageWaitingTime(l) / self.action_t)
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
                                        libsalt.link.getSumTravelTime(l) / (len(link_list_0) * self.reward_info_collection_cycle))
        return list(reward_info)


    #  TL 단위로 보상 관련 정보 수집
    def gatherRewardRelatedInfoPerTL(self, simulation_steps):
        '''
        gather reward related info per TL
        :param action_t: the unit time of green phase allowance (args.action_t)
                        how open do we check for signal change when action is KC(keep or change)
        :param simulation_steps:
        :param reward_info_collection_cycle:
        :return:
        '''
        # from sappo_offset_single.py @ multi-agent/env

        for sa_idx in range(self.num_target):
            sa = self.target_name_list[sa_idx]

            tl_list = self.sa_obj[sa]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                link_list_0 = self.tl_obj[tl_list[tl_idx]]['in_edge_list_0']
                lane_list_0 = self.tl_obj[tl_list[tl_idx]]['in_lane_list_0']

                self.reward_related_info[sa_idx][tl_idx] = \
                    self.__getRewardInfo(self.reward_related_info[sa_idx][tl_idx], link_list_0,
                                                 lane_list_0, simulation_steps)



    def __gatherRewardRelatedInfoGivenTL(self, sa_idx, tl_idx, simulation_steps):
        '''
        gather reward related info for given TL
        :param sa_idx
        :param tl_idx
        :param simulation_steps:
        :return:
        '''
        # from sappo_offset_single.py @ multi-agent/env

        sa = self.target_name_list[sa_idx]

        tl_list = self.sa_obj[sa]['tlid_list']
        link_list_0 = self.tl_obj[tl_list[tl_idx]]['in_edge_list_0']
        lane_list_0 = self.tl_obj[tl_list[tl_idx]]['in_lane_list_0']

        tl_reward_info = []
        tl_reward_info = self.__getRewardInfo(tl_reward_info, link_list_0,
                                                 lane_list_0, simulation_steps)

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
            self.sa_rewards[sa_idx] = np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt':
            self.sa_rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_max':
            self.sa_rewards[sa_idx] = -np.max(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_SBV':
            self.sa_rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_SBV_max':
            self.sa_rewards[sa_idx] = -np.mean(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wt_ABV':
            self.sa_rewards[sa_idx] = -np.mean(self.reward_related_info[sa_idx])

        elif self.reward_func == 'wq' or self.reward_func == 'cwq':
            self.sa_rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx]) / 5000
        elif self.reward_func == 'wq_median':
            if len(self.reward_related_info[sa_idx]) == 0:
                self.sa_rewards[sa_idx] = 0
            else:
                self.reward_related_info[sa_idx][self.reward_related_info[sa_idx] == 0] = np.nan
                if np.isnan(np.nanmedian(self.reward_related_info[sa_idx])):
                    self.sa_rewards[sa_idx] = 0
                else:
                    self.sa_rewards[sa_idx] = -np.nanmedian(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wq_min':
            self.sa_rewards[sa_idx] = -np.min(self.reward_related_info[sa_idx])
        elif self.reward_func == 'wq_max':
            if len(self.reward_related_info[sa_idx]) == 0:
                self.sa_rewards[sa_idx] = 0
            else:
                self.sa_rewards[sa_idx] = -np.max(self.reward_related_info[sa_idx])

        elif self.reward_func == 'tt':
            self.sa_rewards[sa_idx] = -np.sum(self.reward_related_info[sa_idx])

        # clear gathered reward related info
        # cumulate the reward related info when reward func is cwq
        if self.reward_func != 'cwq':
            self.reward_related_info[sa_idx].clear()
            # todo AttributeError: 'numpy.ndarray' object has no attribute 'clear'
            # self.reward_related_info[sa_idx] = []

    def calculateRewardPerSA_V2(self, sa_idx):
        op_str = self.DIC_CALC_REWARD[self.reward_func][0]
        op_func = self.DIC_OP_FUNC[op_str]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]

        self.sa_rewards[sa_idx] = op_func(self.reward_related_info[sa_idx])* weight

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
    #     sa = self.target_name_list[sa_idx]
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
        op_func = self.DIC_OP_FUNC[op_str]
        sa_reward = 0  # for self.rewards[sa_idx]
        tl_reward = 0  # for self.tl_reward_dic[tlid]
        sa = self.target_name_list[sa_idx]

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

        self.sa_rewards[sa_idx] = sa_reward


    def __innerCalculateRewardPerTL_V2(self, sa_idx, op_str, weight):
        op_func = self.DIC_OP_FUNC[op_str]
        # sa_reward = 0  # for self.rewards[sa_idx]
        # tl_reward = 0  # for self.tl_reward_dic[tlid]
        sa = self.target_name_list[sa_idx]

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

        self.sa_rewards[sa_idx] = sa_reward


        if DBG_OPTIONS.PrintRewardMgmt:
            print(f'{sa} : {self.sa_rewards[sa_idx]}')

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
        self.sa_rewards = 0
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
            sa_name = self.target_name_list[sa_idx]
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
            sa_name = self.target_name_list[sa_idx]
            tl_list = self.sa_obj[sa_name]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                self.reward_related_info[sa_idx][tl_idx].clear()



    # todo delete comment : 특정 교차로의 보상을 계산한다.
    def calculateTLRewardInstantly(self, sa_idx, tlid, sim_step):
        '''
        calculate rewards for given TL
        gather reward related info instantly

        :param sa_idx:
        :param tlid:
        :param action_t: the unit time of green phase allowance (args.action_t)
                        how open do we check for signal change when action is KC(keep or change)
        :param sim_step:
        :param reward_info_collection_cycle:  SappoEnv.py 에있는 상수 이다. args 혹은 TRAIN_CONFIG에 넣어야 할 것 같다.
        :return:
        '''
        op_str = self.DIC_CALC_REWARD[self.reward_func][0]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]
        op_func = self.DIC_OP_FUNC[op_str]
        sa = self.target_name_list[sa_idx]
        tl_list = self.sa_obj[sa]['tlid_list']

        tl_idx = tl_list.index(tlid)

        #todo : do you think which one is proper?
        if 0:
            tl_reward_info = self.reward_related_info[sa_idx][tl_idx]
        else:
            tl_reward_info = self.__gatherRewardRelatedInfoGivenTL(sa_idx, tl_idx, sim_step)

        if len(tl_reward_info) == 0:
            tl_reward = 0
        else:
            tl_reward = op_func(tl_reward_info) * weight

        return tl_reward



# self.reward_unit 에 따라 다르게 수집하게 하자.
class SaltRewardMgmtV3:
    '''
    class for reward management
     - gather info which will be used to calculate reward
     - calculate reward
    '''
    def __init__(self, reward_func, gather_unit, action_t, reward_info_collection_cycle, sa_obj, tl_obj, target_name_list, num_target=-1):
        '''
        constructor
        :param reward_func:
        :param gather_unit:
        :param action_t: the unit time of green phase allowance (args.action_t)
                        how open do we check for signal change when action is KC(keep or change)
        :param reward_info_collection_cycle : Information collection cycle for reward calculation
        :param sa_obj: object which contains information about SAs
        :param tl_obj: object which contains information about TLs
        :param target_name_list:
        :param num_target:
        '''

        self.DIC_CALC_REWARD, self.DIC_OP_FUNC = self.__constructDictionaryForRewardCalculation()

        if reward_func not in self.DIC_CALC_REWARD.keys():
            raise Exception("internal error : should check dic_calc_reward")

        self.reward_func = reward_func
        self.reward_unit = gather_unit.upper()  # _REWARD_GATHER_UNIT_
        self.action_t = action_t
        self.reward_info_collection_cycle = reward_info_collection_cycle
        self.sa_obj = sa_obj # object which holds information for all SA
        self.tl_obj = tl_obj # object which holds information for all TL
        self.target_name_list = target_name_list  # SA name list
        self.num_target = len(target_name_list) if num_target == -1 else num_target
        # reward is only needed to train/test target SA

        self.agent_num = len(self.target_name_list)

        # to save rewards for SA
        self.sa_rewards = list(0 for i in range(self.agent_num))

        self.reward_related_info = []
        self.tl_reward_dic = {}  # reward for TL

        for sa_idx in range(self.num_target):
            sa_reward_info = []
            sa_name = self.target_name_list[sa_idx]
            tl_list = self.sa_obj[sa_name]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                if self.reward_unit == _REWARD_GATHER_UNIT_.TL:
                    sa_reward_info.append([])
                self.tl_reward_dic[tl_list[tl_idx]] = 0

            self.reward_related_info.append(sa_reward_info)
            # reward_related_info = [
            #            # add [ [],..., [] ] for SA_a if REWARD_GATHER_UNIT is TL : an inner list per TL
            #             [ [tl1's link_1...tl1's link_n], ... , [tl_n's link_1...tl_n's link_n] ]
            #
            #            #  only add [] for SA_b if REWARD_GATHER_UNIT is SA : all thing are in a list
            #             [ link_1...link_n ]
            #       ]


    def __constructDictionaryForRewardCalculation(self):
        '''
        construct two dictionary for reward calculation : DIC_CALC_REWARD, DIC_OP_FUNC
            dic_calc_reward : mapping dictionary : reward func --> {operation, weight}
                key : reward function
                value : [operation, weight] pairs to be used to calculate reward
            dic_op_func : mapping dictionary : operation --> function
                key : operation
                value : function to be used to calculate reward
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

        dic_op_func = {}
        dic_op_func['sum'] = np.sum
        dic_op_func['max'] = np.max
        dic_op_func['min'] = np.min
        dic_op_func['mean'] = np.mean

        return dic_calc_reward, dic_op_func



    def reset(self):
        '''
        reset objects to save reward related info : self.reward_related_info, self.tl_reward_dic, self.sa_rewards
        :return:
        '''
        for sa_idx in range(self.num_target):
            sa_name = self.target_name_list[sa_idx]
            tl_list = self.sa_obj[sa_name]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                if self.reward_unit == _REWARD_GATHER_UNIT_.TL:
                    self.reward_related_info[sa_idx][tl_idx].clear()
                else:
                    self.reward_related_info[sa_idx].clear()
                self.tl_reward_dic[tl_list[tl_idx]] = 0
            self.sa_rewards[sa_idx] = 0



    def gatherRewardRelatedInfo(self, simulation_steps):
        '''
        gather reward related info

        :param simulation_steps:
        :return:
        '''
        if self.reward_unit == _REWARD_GATHER_UNIT_.SA:
            return self.__gatherRewardRelatedInfoPerSA(simulation_steps)
        else: # self.reward_unit == _REWARD_GATHER_UNIT_.TL
            return self.__gatherRewardRelatedInfoPerTL(simulation_steps)

    def __gatherRewardRelatedInfoPerSA(self, simulation_steps):
        '''
        gather reward related info
        all info about same SA is integrated(combined) into a list
        :param simulation_steps:
        :return:
        '''
        # from sappo_offset_single.py @ multi-agent/env

        for sa_idx in range(self.num_target):
            sa = self.target_name_list[sa_idx]
            link_list_0 = self.sa_obj[sa]['in_edge_list_0']
            lane_list_0 = self.sa_obj[sa]['in_lane_list_0']
            # link_list_1 = self.sa_obj[sa]['in_edge_list_1']
            # lane_list_1 = self.sa_obj[sa]['in_lane_list_1']

            self.reward_related_info[sa_idx] = self.__getRewardInfo(self.reward_related_info[sa_idx], link_list_0,
                                                                    lane_list_0, simulation_steps)



    def __getRewardInfo(self, reward_info, link_list_0, lane_list_0, sim_steps):
        '''
        gather info which will be used to calculate reward
        and combine gathered info with previously collected info

        todo : should avoid using constants

        :param reward_info: previously collected info
        :param link_list_0: list of zero-hop link
        :param lane_list_0: list of zero-hop lane
        :param sim_steps: current simulation step
        :return:
        '''

        if self.reward_func == 'pn':
            for l in link_list_0:
                reward_info = np.append(reward_info, libsalt.link.getSumPassed(l))

        elif self.reward_func in ['wt', 'wt_max']:
            for l in link_list_0:
                reward_info = np.append(reward_info, libsalt.link.getAverageWaitingTime(l) / self.action_t)
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
                                        libsalt.link.getSumTravelTime(l) / (len(link_list_0) * self.reward_info_collection_cycle))
        return list(reward_info)



    def __gatherRewardRelatedInfoPerTL(self, simulation_steps):
        '''
        gather reward related info per TL
        all info about same TL is integrated(combined) into a list

        :param simulation_steps:
        :return:
        '''
        # from sappo_offset_single.py @ multi-agent/env

        for sa_idx in range(self.num_target):
            sa = self.target_name_list[sa_idx]

            tl_list = self.sa_obj[sa]['tlid_list']
            num_tl = len(tl_list)
            for tl_idx in range(num_tl):
                link_list_0 = self.tl_obj[tl_list[tl_idx]]['in_edge_list_0']
                lane_list_0 = self.tl_obj[tl_list[tl_idx]]['in_lane_list_0']

                self.reward_related_info[sa_idx][tl_idx] = \
                    self.__getRewardInfo(self.reward_related_info[sa_idx][tl_idx], link_list_0,
                                                 lane_list_0, simulation_steps)





    def __calculateRewardPerSA(self, sa_idx):
        '''
        calculate reward about given SA
        :param sa_idx:
        :return:
        '''
        op_str = self.DIC_CALC_REWARD[self.reward_func][0]
        op_func = self.DIC_OP_FUNC[op_str]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]

        self.sa_rewards[sa_idx] = op_func(self.reward_related_info[sa_idx])* weight

        # clear gathered reward related info
        # cumulate the reward related info when reward func is cwq
        if self.reward_func != 'cwq':
            self.reward_related_info[sa_idx].clear()



    def __calculateRewardPerTL(self, sa_idx):
        '''
        calculate reward about TLs which are belong to given SA
        also calculate reward for given SA
        :param sa_idx:
        :return:
        '''
        # todo     env/sappo_noConst.py : args.cp 관련 정리 필요
        #

        op_str = self.DIC_CALC_REWARD[self.reward_func][0]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]

        op_func = self.DIC_OP_FUNC[op_str]
        sa_name = self.target_name_list[sa_idx]

        tl_list = self.sa_obj[sa_name]['tlid_list']
        num_tl = len(tl_list)

        # calculate reward for TLs which are belongs to given SA
        # and integrate reward related info about TL to be used to calculate reward for a given SA
        sa_info = []
        for tl_idx in range(num_tl):
            tlid = tl_list[tl_idx]
            if len(self.reward_related_info[sa_idx][tl_idx]) == 0:
                tl_reward = 0
            else:
                tl_reward = op_func(self.reward_related_info[sa_idx][tl_idx]) * weight
            self.tl_reward_dic[tlid] = tl_reward
            sa_info += self.reward_related_info[sa_idx][tl_idx]


        # calculate reward for a SA
        self.sa_rewards[sa_idx] = op_func(sa_info) * weight

        # dump calculated reward
        if DBG_OPTIONS.PrintRewardMgmt:
            print(f'{sa_name} : {self.sa_rewards[sa_idx]}')

            for tl_idx in range(num_tl):
                tlid = tl_list[tl_idx]
                print(f"\t{tlid} : {self.tl_reward_dic[tlid]}")


        # clear gathered reward related info
        # cumulate the reward related info when reward func is cwq
        if self.reward_func != 'cwq':
            for tl_idx in range(num_tl):
                self.reward_related_info[sa_idx][tl_idx].clear()



    def calculateReward(self, sa_idx):
        '''
        calculate rewards for a SA indicated by given sa_idx
        :param sa_idx:
        :return:
        '''

        assert sa_idx < self.num_target, f"internal error : SA indicated by sa_idx({sa_idx}) is not target of training : all SA={self.sa_obj.keys()}"

        if self.reward_unit == _REWARD_GATHER_UNIT_.SA:
            self.__calculateRewardPerSA(sa_idx)
        else: # self.reward_unit == _REWARD_GATHER_UNIT_.TL
            self.__calculateRewardPerTL(sa_idx)




    def __gatherRewardRelatedInfoGivenTL(self, sa_idx, tl_idx, simulation_step):
        '''
        gather reward related info given TL at given simulation step; not accumulated
        :param sa_idx  : index of SA
        :param tl_idx : index(location) of TL in a given SA
        :param simulation_steps:
        :return:
        '''

        sa = self.target_name_list[sa_idx]

        tl_list = self.sa_obj[sa]['tlid_list']
        link_list_0 = self.tl_obj[tl_list[tl_idx]]['in_edge_list_0']
        lane_list_0 = self.tl_obj[tl_list[tl_idx]]['in_lane_list_0']

        tl_reward_info = []
        tl_reward_info = self.__getRewardInfo(tl_reward_info, link_list_0, lane_list_0, simulation_step)

        return tl_reward_info



    def calculateTLRewardInstantly(self, sa_idx, tlid, sim_step):
        '''
        calculate rewards for given TL instantly
        gather reward related info instantly and calculate reward

        :param sa_idx:
        :param tlid:
        :param sim_step:
        :return:
        '''
        op_str = self.DIC_CALC_REWARD[self.reward_func][0]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]
        op_func = self.DIC_OP_FUNC[op_str]
        sa = self.target_name_list[sa_idx]
        tl_list = self.sa_obj[sa]['tlid_list']

        tl_idx = tl_list.index(tlid)

        tl_reward_info = self.__gatherRewardRelatedInfoGivenTL(sa_idx, tl_idx, sim_step)

        if len(tl_reward_info) == 0:
            tl_reward = 0
        else:
            tl_reward = op_func(tl_reward_info) * weight

        return tl_reward



    def __gatherRewardRelatedInfoGivenSA(self, sa_idx, sim_steps):
        '''
        gather reward related info for given SA
        :param sa_idx
        :param sim_steps:
        :return:
        '''

        sa = self.target_name_list[sa_idx]
        link_list_0 = self.sa_obj[sa]['in_edge_list_0']
        lane_list_0 = self.sa_obj[sa]['in_lane_list_0']

        sa_reward_info = []
        sa_reward_info = self.__getRewardInfo(sa_reward_info, link_list_0, lane_list_0, sim_steps)

        return sa_reward_info



    def calculateSARewardInstantly(self, sa_idx, sim_step):
        '''
        calculate rewards for given SA instantly
        gather reward related info instantly and calculate reward

        :param sa_idx:
        :param sim_step:
        :return:
        '''
        op_str = self.DIC_CALC_REWARD[self.reward_func][0]
        weight = self.DIC_CALC_REWARD[self.reward_func][1]
        op_func = self.DIC_OP_FUNC[op_str]
        sa = self.target_name_list[sa_idx]

        sa_reward_info = self.__gatherRewardRelatedInfoGivenSA(sa_idx, sim_step)

        if len(sa_reward_info) == 0:
            sa_reward = 0
        else:
            sa_reward = op_func(sa_reward_info) * weight

        return sa_reward