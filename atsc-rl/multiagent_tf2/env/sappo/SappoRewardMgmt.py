# -*- coding: utf-8 -*-

import numpy as np

from DebugConfiguration import DBG_OPTIONS
from TSOConstants import _REWARD_GATHER_UNIT_
from TSOConstants import _RESULT_COMP_
from TSOUtil import getTsoOutputInfo
from TSOUtil import replaceTsoOutputInfo


class SappoRewardMgmt:
    '''
    class for SAPPO reward management
     - gather info which will be used to calculate reward
     - calculate reward

     exclude  wt_SBV, wt_SBV_max and wt_ABV from reward functions because it is TOO SLOW.
      (SBV - sum-based, ABV - average-base)

    '''
    def __init__(self, te_conn, reward_func, gather_unit, action_t, reward_info_collection_cycle, sa_obj, tl_obj, target_name_list, num_target=-1):
        '''
        constructor
        :param te_conn
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

        self.te_conn = te_conn
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
        # dic_calc_reward['wt_ABV'] = ['mean', -1]
        dic_calc_reward['wt_median'] = ['median', -1]
        dic_calc_reward['wt_max'] = ['max', -1]
        # dic_calc_reward['wt_SBV'] = ['sum', -1]
        # dic_calc_reward['wt_SBV_max'] = ['max', -1]

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
            self.reward_related_info[sa_idx] = self.__getRewardInfo(self.reward_related_info[sa_idx], self.sa_obj[sa])



    def __getRewardInfo(self, reward_info, an_info_dic):
        '''
        gather info which will be used to calculate reward
        and combine gathered info with previously collected info

        :param reward_info: previously collected info
        :param an_info_dic: a dictionary object which holds information about a SA or a TL
        :return:
        '''

        if self.reward_func == 'pn':
            reward_info = self.te_conn.getSumPassed(reward_info, an_info_dic)

        elif self.reward_func in ['wt', 'wt_max']:
            reward_info = self.te_conn.getWaitingTime(reward_info, an_info_dic,  self.action_t)

        elif self.reward_func in ['wq', 'wq_median', 'wq_min', 'wq_max', 'cwq']:
            reward_info = self.te_conn.getWaitingQLength(reward_info, an_info_dic)

        elif self.reward_func == 'tt':
            reward_info = self.te_conn.getTravelTime(reward_info, an_info_dic,  self.reward_info_collection_cycle)

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
                self.reward_related_info[sa_idx][tl_idx] = \
                        self.__getRewardInfo(self.reward_related_info[sa_idx][tl_idx], self.tl_obj[tl_list[tl_idx]])




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

        tl_reward_info = []
        tl_reward_info = self.__getRewardInfo(tl_reward_info, self.tl_obj[tl_list[tl_idx]])

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

        sa_reward_info = []
        sa_reward_info = self.__getRewardInfo(sa_reward_info, self.sa_obj[sa])

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



    def appendPhaseRewards(self, fn, sim_step, actions, sa_obj, sa_name_list, tl_obj, tl_id_list, tso_output_info_dic):
        '''
            write reward to given file
            this func is called in TEST-, SIMULATE-mode to write reward info which will be used by visualization tool

            :param fn: file name to store reward
            :param sim_step: simulation step
            :param actions: applied actions
            :param sa_obj: object which holds information about SAs
            :param sa_name_list:  list of name of SA
            :param tl_obj: object which holds information about TLs
            :param tl_id_list: list of TL id
            :param tso_output_info_dic: dictionary which holds TSO-relates output information such as average speed, average travel time, sum travel time, num of passed vehicles
            :return:
            '''

        f = open(fn, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None, closefd=True, opener=None)

        num_target = len(sa_name_list)
        if self.reward_unit == _REWARD_GATHER_UNIT_.SA:
            sa_reward_list = []
            for sa_idx in range(num_target):
                sa_reward = self.calculateSARewardInstantly(sa_idx, sim_step)
                sa_reward_list.append(sa_reward)

            for i in range(len(tl_id_list)):
                tlid = tl_id_list[i]

                sa_name = tl_obj[tlid]['signalGroup']
                sa_idx = sa_name_list.index(sa_name)

                reward = sa_reward_list[sa_idx]

                tl_action = 0
                if len(actions) != 0:
                    tl_idx = sa_obj[sa_name]['tlid_list'].index(tlid)
                    tl_action = actions[sa_idx][tl_idx]

                if (sim_step % _RESULT_COMP_.SPEED_GATHER_INTERVAL) == 0:
                    avg_speed, avg_tt, sum_passed, sum_travel_time = self.te_conn.gatherTsoOutputInfo(tlid, tl_obj, num_hop=0)
                    tso_output_info_dic = replaceTsoOutputInfo(tso_output_info_dic, i, avg_speed, avg_tt, sum_passed,
                                                               sum_travel_time)

                avg_speed, avg_tt, sum_passed, sum_travel_time, offset, duration = getTsoOutputInfo(tso_output_info_dic, i)

                tl_action = f'{tl_action}#{offset}#{duration}'
                current_phase_index = self.te_conn.getCurrentPhaseIndex(tlid)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(sim_step,
                                                              tl_obj[tlid]['crossName'],
                                                              tl_action,
                                                              current_phase_index,
                                                              reward,
                                                              avg_speed,
                                                              avg_tt,
                                                              sum_passed,
                                                              sum_travel_time))

            sa_reward_list.clear()

        else:  # reward_mgmt.reward_unit == _REWARD_GATHER_UNIT_.TL
            for i in range(len(tl_id_list)):
                tlid = tl_id_list[i]

                sa_name = tl_obj[tlid]['signalGroup']
                sa_idx = sa_name_list.index(sa_name)

                reward = self.calculateTLRewardInstantly(sa_idx, tlid, sim_step)

                tl_action = 0
                if len(actions) != 0:
                    tl_idx = sa_obj[sa_name]['tlid_list'].index(tlid)
                    tl_action = actions[sa_idx][tl_idx]

                if (sim_step % _RESULT_COMP_.SPEED_GATHER_INTERVAL) == 0:
                    avg_speed, avg_tt, sum_passed, sum_travel_time = self.te_conn.gatherTsoOutputInfo(tlid, tl_obj, num_hop=0)
                    tso_output_info_dic = replaceTsoOutputInfo(tso_output_info_dic, i, avg_speed, avg_tt, sum_passed,
                                                               sum_travel_time)
                avg_speed, avg_tt, sum_passed, sum_travel_time, offset, duration  = getTsoOutputInfo(tso_output_info_dic, i)

                tl_action = f'{tl_action}#{offset}#{duration}'

                current_phase_index = self.te_conn.getCurrentPhaseIndex(tlid)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(sim_step,
                                                              tl_obj[tlid]['crossName'],
                                                              tl_action,
                                                              current_phase_index,
                                                              reward,
                                                              avg_speed,
                                                              avg_tt,
                                                              sum_passed,
                                                              sum_travel_time))

        f.close()

