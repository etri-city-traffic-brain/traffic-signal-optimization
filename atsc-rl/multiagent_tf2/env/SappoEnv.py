# -*- coding: utf-8 -*-

import gym
from gym import spaces
import os
import sys
import numpy as np


import libsalt

from DebugConfiguration import DBG_OPTIONS


from env.SaltEnvUtil import appendPhaseRewards, gatherTsoOutputInfo
from env.SaltEnvUtil import appendTsoOutputInfo, initTsoOutputInfo

if DBG_OPTIONS.RichActionOutput:
    from env.SaltEnvUtil import appendTsoOutputInfoSignal
    from env.SaltEnvUtil import replaceTsoOutputInfoOffset, replaceTsoOutputInfoDuration


from env.SaltEnvUtil import copyScenarioFiles
from env.SaltEnvUtil import getSaRelatedInfo
from env.SaltEnvUtil import getSimulationStartStepAndEndStep
from env.SaltEnvUtil import makePosssibleSaNameList
from env.SappoActionMgmt import SaltActionMgmt
from env.SappoRewardMgmt import _REWARD_GATHER_UNIT_, SaltRewardMgmtV3
from TSOUtil import writeLine
from TSOUtil import getOutputDirectoryRoot



class SaltSappoEnvV3(gym.Env):
    '''
    a class for Sappo Environment
    : previously it is called SALT_SAPPO_20220420
       - considered infer_TL
       - separate action/reward mgmt as a class
       - supported action : offset, gr(green ratio), gro(green ratio + offset), kc(keep or change)
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, args):  # equal SALT_SAPPO_offset_single.__init__()
        '''
        constructor
        :param args:
        '''
        self.env_name = "SALT_SAPPO"

        # check environment
        if 'SALT_HOME' in os.environ:
            tools = os.path.join(os.environ['SALT_HOME'], 'tools')
            sys.path.append(tools)

            tools_libsalt = os.path.join(os.environ['SALT_HOME'], 'tools/libsalt')
            sys.path.append(tools_libsalt)
        else:
            sys.exit("Please declare the environment variable 'SALT_HOME'")

        # initialize
        if 1:
            ##-- 멤버 속성 초기화 (설정)
            # self.reward_func = args.reward_func
            self.action_t = args.action_t
            self.args = args
            # self.cp = args.cp  # [in KC] action change penalty ... currently not used
            self.print_out = args.print_out  # 각 Step에서 결과 출력 : action, phase, reward

            # #todo check reasons... why it takes long time if we user TRAIN_CONFIG...
            # self.state_weight = TRAIN_CONFIG['state_weight']   # not used
            # self.reward_weight = TRAIN_CONFIG['reward_weight']  # not used
            # self.sim_period = TRAIN_CONFIG['sim_period']  # 보상 계산을 위한 정보 수집 주기
            #
            # self.state_weight = state_weight  # not used
            # self.reward_weight = reward_weight  # not used
            self.reward_info_collection_cycle = args.reward_info_collection_cycle  # 보상 계산을 위한 정보 수집 주기, previously it was 'sim_period'

            self.warming_up_time = args.warmup_time # simulation warming up
            self.control_cycle = args.control_cycle  # how open change offset : ex., every 5 cycle

        # calculate start-/end-time of simulation
        self.start_step, self.end_step = getSimulationStartStepAndEndStep(args)

        ## copy scenario related files and get copied scenario file path
        self.salt_scenario = copyScenarioFiles(args.scenario_file_path)

        # gather information related to the intersection to be optimized from scenario file : TL, TSS, lane, link, ...
        if 1:
            possible_target_sa_name_list = makePosssibleSaNameList(args.target_TL)

            target_tl_obj, target_sa_obj, _lane_len = \
                getSaRelatedInfo(args, possible_target_sa_name_list, self.salt_scenario)

            self.HAVE_INFER_TL = False

            if self.args.infer_TL.strip() != "":
                self.HAVE_INFER_TL = True

            infer_sa_obj = {}
            infer_tl_obj = {}

            if self.HAVE_INFER_TL:
                possible_infer_sa_name_list = makePosssibleSaNameList(args.infer_TL)

                infer_tl_obj, infer_sa_obj, _lane_len = \
                    getSaRelatedInfo(args, possible_infer_sa_name_list, self.salt_scenario)

            ##--- construct SA obj dictionary
            self.sa_obj = {} # 전체 SA obj 저장
            self.sa_obj.update(target_sa_obj)
            self.sa_obj.update(infer_sa_obj)

            ##-- set # of agent : an agent per SA
            self.agent_num = len(self.sa_obj)
            self.train_agent_num = len(target_sa_obj)

            ##--- construct TL obj dictionary
            self.tl_obj = {}  # 전체 TL obj 저장
            self.tl_obj.update(target_tl_obj)
            self.tl_obj.update(infer_tl_obj)


            ##--- construct TL id list
            self.target_tl_id_list = list(target_tl_obj.keys())
            infer_tl_id_list = list(infer_tl_obj.keys())
            self.tl_id_list = self.target_tl_id_list + infer_tl_id_list

            ##-- initialize so that information on one SA can be accessed with the same index
            ##         동일한 인덱스로 하나의 SA에 대한 정보에 접근 가능하도록 초기화
            ##      sa_name_list : SA name
            ##      sa_cycle_list : Cycle Length of SA
            ##      time_to_act_list : time to get next action

            ### cosntruct sa_name_list, sa_cycle_list, time_to_act_list
            self.target_sa_name_list = list(target_sa_obj.keys())  # name of SA
            infer_sa_name_list = list(infer_sa_obj.keys())
            self.sa_name_list = self.target_sa_name_list + infer_sa_name_list

            self.sa_cycle_list = []     # cycle length
            self.time_to_act_list = []  # 다음 번 모델 추론을 할 time step

            for sa_name in self.sa_name_list:
                sa_cycle_length = self.sa_obj[sa_name]['cycle_list'][0]
                # -- cycle length of SAs
                self.sa_cycle_list.append(sa_cycle_length)
                # -- time to act of SAs : should larger than warming_up_time
                next_time_to_act = self.__getNextTimeToAct(self.warming_up_time, sa_cycle_length, self.control_cycle)
                self.time_to_act_list.append(next_time_to_act)

            ##-- initialize reward related things
            ##   reward mgmt : gather the reward related information and calculate reward
            ##   only needed to train/test target SA
            # self.reward_mgmt = SaltRewardMgmt(args.reward_func, areg.reward_gather_unit, self.sa_obj, self.sa_name_list, len(self.target_sa_name_list))
            # self.reward_mgmt = SaltRewardMgmtV2(args.reward_func, gather_unit, self.action_t, self.reward_info_collection_cycle, self.sa_obj, self.tl_obj, self.sa_name_list, len(self.target_sa_name_list))
            self.reward_mgmt = SaltRewardMgmtV3(args.reward_func, args.reward_gather_unit, self.action_t, self.reward_info_collection_cycle, self.sa_obj, self.tl_obj, self.sa_name_list, len(self.target_sa_name_list))

            ##-- initialize action related things
            ##   action mgmt : make phase array, convert model output into discrete action, apply action to env
            ##   all SA need action (mgmt)
            self.action_mgmt = SaltActionMgmt(self.args, self.sa_obj, self.sa_name_list)

            # Index for SA where action must be determined through model inference
            self.idx_of_act_sa = []

            ##-- initialize state(observation) related things
            ##   ALl SA need State(Observation)
            self.observations = []
            for sa_name in self.sa_name_list:
                self.observations.append([0] * self.sa_obj[sa_name]['state_space'])

            self.simulation_steps = 0

            # initialize discrete actions
            self.discrete_actions = list()

            # dictionary to hold TSO output information
            #   : will be dumped into TSO output file(rl_phase_reward_output.txt)
            self.tso_output_info_dic = initTsoOutputInfo()

            if self.args.mode == 'test':
                self.fn_rl_phase_reward_output = "{}/output/test/rl_phase_reward_output.txt".format(getOutputDirectoryRoot(args))

                writeLine(self.fn_rl_phase_reward_output,
                          'step,tl_name,actions,phase,reward,avg_speed,avg_travel_time,sum_passed,sum_travel_time')



    def __getNextTimeToAct(self, current_step, sa_cycle, control_cycle):
        '''
        get next time to act
        "act" means infer to get action and apply this action

        :param current_step: current simulation step
        :param sa_cycle: cycle length of SA
        :param control_cycle:
        :return:
        '''

        if self.args.action in set(["offset", "gr", "gro"]):
            interval = sa_cycle * control_cycle
            last_num = int(current_step / interval)  # 버림
            next_time_to_act = (last_num + 1) * interval
        elif self.args.action == "kc":
            next_time_to_act = current_step + 1 # self.action_t?

        return next_time_to_act


    # todo : state 관련된 것들도 하나의 클래스로 묶으면 좋을 것 같다....
    #                get_obj(), getState()에 분산되어 있다.
    def __getState(self, an_sa_obj, tl_objs):
        if DBG_OPTIONS.MergeAfterNormalize:
            return self.__getState_V2(an_sa_obj, tl_objs)
        else:
            return self.__getState_V1(an_sa_obj, tl_objs)

    def __getState_V1(self, an_sa_obj, tl_objs):

        '''
        gather state information of given SA

        :param an_sa_obj:
        :param tl_objs:
        :return:
        '''
        obs = []
        densityMatrix = []
        passedMatrix = []
        vddMatrix = []
        tlMatrix = []

        for tlid in an_sa_obj['tlid_list']:
            lane_list = tl_objs[tlid]['in_lane_list']
            lane_list_0 = tl_objs[tlid]['in_lane_list_0']

            for lane in lane_list_0:
                if self.args.state == 'd':
                    densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(lane))
                if self.args.state == 'v':
                    passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(lane))
                if self.args.state == 'vd':
                    densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(lane))
                    passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(lane))
                if self.args.state == 'vdd':
                    vddMatrix = np.append(vddMatrix, libsalt.lane.getNumVehPassed(lane) / (
                                libsalt.lane.getAverageDensity(lane) + sys.float_info.epsilon))

            tlMatrix = np.append(tlMatrix, libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid))

            if self.args.state == 'd':
                obs = np.append(densityMatrix, tlMatrix)
            if self.args.state == 'v':
                obs = np.append(passedMatrix, tlMatrix)
            if self.args.state == 'vd':
                obs = np.append(densityMatrix, passedMatrix)
                obs = np.append(obs, tlMatrix)
            if self.args.state == 'vdd':
                obs = np.append(vddMatrix, tlMatrix)

        # normalize : 0 .. 1
        # todo think : Is there any possibility of side effect when value of one fild is large
        #            .... density인 경우에는 tlMatrix의 값이 너무 크지 않은가?
        #            .... num passed vehicle이 포함된 경우에는 이 값이 너무 크지 않은가?
        #            .... 수집 주기가 cycle length 라면 항상 phaseIdx는 같은 값이지 않나?
        #             각각에 대해 정규화 후에 합치는 것은 하나의 대안이 될 수 있다?  ref. __getState_V2()
        if DBG_OPTIONS.DoNormalize:
            obs = obs + np.finfo(float).eps
            obs = obs / np.max(obs)

        return obs

    def __normalize(self, obs_matrix):
        if DBG_OPTIONS.DoNormalize:
            obs_matrix = obs_matrix + np.finfo(float).eps
            obs_matrix = obs_matrix / np.max(obs_matrix)
        return obs_matrix

    def __getState_V2(self, an_sa_obj, tl_objs):
        '''
        gather state information of given SA

        difference from V1 : merge after normalization

        :param an_sa_obj:
        :param tl_objs:
        :return:
        '''


        obs = []
        density_matrix = []
        passed_matrix = []
        vdd_matrix = []
        tl_matrix = []

        max_num_phase = 0

        for tlid in an_sa_obj['tlid_list']:
            cur_tl_num_phase = len(tl_objs[tlid]['duration'])
            if max_num_phase < cur_tl_num_phase:
                max_num_phase = cur_tl_num_phase

            # print(f'tlid={tlid} cur_tl_num_phase={cur_tl_num_phase}  max_num_phase = {max_num_phase}')
            lane_list = tl_objs[tlid]['in_lane_list']
            lane_list_0 = tl_objs[tlid]['in_lane_list_0']

            for lane in lane_list_0:
                if self.args.state == 'd':
                    density_matrix = np.append(density_matrix, libsalt.lane.getAverageDensity(lane))
                if self.args.state == 'v':
                    passed_matrix = np.append(passed_matrix, libsalt.lane.getNumVehPassed(lane))
                if self.args.state == 'vd':
                    density_matrix = np.append(density_matrix, libsalt.lane.getAverageDensity(lane))
                    passed_matrix = np.append(passed_matrix, libsalt.lane.getNumVehPassed(lane))
                if self.args.state == 'vdd':
                    vdd_matrix = np.append(vdd_matrix, libsalt.lane.getNumVehPassed(lane) / (
                                libsalt.lane.getAverageDensity(lane) + sys.float_info.epsilon))

            tl_matrix = np.append(tl_matrix, libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid))

        # tl_matrix = self.__normailze(tl_matrix)
        tl_matrix = tl_matrix/(max_num_phase - 1)  # normalize

        if self.args.state == 'd':
            density_matrix = self.__normalize(density_matrix)
            obs = np.append(density_matrix, tl_matrix)
        if self.args.state == 'v':
            passed_matrix = self.__normalize(passed_matrix)
            obs = np.append(passed_matrix, tl_matrix)
        if self.args.state == 'vd':
            density_matrix = self.__normalize(density_matrix)
            passed_matrix = self.__normalize(passed_matrix)
            obs = np.append(density_matrix, passed_matrix)
            obs = np.append(obs, tl_matrix)
        if self.args.state == 'vdd':
            vdd_matrix = self.__normalize(vdd_matrix)
            obs = np.append(vdd_matrix, tl_matrix)

        return obs




    def __getTimeToActInfo(self):
        '''
        get info(time and index of SA) to determine action through model inference (next_act_time, idx_of_next_act_time)
        :return:
        '''
        next_act_time = min(self.time_to_act_list)
        idx_of_next_act_time = list(filter(lambda x:
                                           self.time_to_act_list[x] == next_act_time,
                                           range(len(self.time_to_act_list))))
        return next_act_time, idx_of_next_act_time



    def step(self, actions):
        '''
        apply actions
        and gather rewards & observations

        :param actions:
        :return:
        '''
        self.done = False

        # change phase array by applying action
        #-- action에 따라 신호 페이즈 집합(self.action_mgmt.apply_phase_array_list)을 변경한다.
        #   지난 번 step에서 새로운 action을 적용할 시간이 도래한 것들에 대해 action 적용하여 변경
        for i in self.idx_of_act_sa:
            ###-- convert action : i.e., make discrete action
            sa_name = self.sa_name_list[i]
            discrete_action = self.action_mgmt.convertToDiscreteAction(sa_name, actions[i])
            self.discrete_actions[i] = discrete_action

            if DBG_OPTIONS.PrintAction:
                print(f"DBG in SappoEnv.step() discrete_actions_{i}={discrete_action}")

            if DBG_OPTIONS.RichActionOutput:
                # offset_list, duration_list = self.action_mgmt.changePhaseArray(self.simulation_steps, i, actions[i])
                offset_list, duration_list = self.action_mgmt.changePhaseArray(self.simulation_steps, i, self.discrete_actions[i])

                if self.args.mode=='test':
                    sa_name = self.sa_name_list[i]
                    an_sa_obj = self.sa_obj[sa_name]
                    an_sa_tlid_list = an_sa_obj['tlid_list']

                    if len(offset_list):
                        # if DBG_OPTIONS.PrintAction:
                        #     print(f'DBG offset_list_{i}={offset_list} changed')

                        for j in range(len(offset_list)):
                            tlid = an_sa_tlid_list[j]
                            ith = self.target_tl_id_list.index(tlid)
                            assert ith < len(self.tso_output_info_dic["offset"]), print(f'ith={ith} len(self.tso_output_info_dic["offset"])={len(self.tso_output_info_dic["offset"])}')
                            replaceTsoOutputInfoOffset(self.tso_output_info_dic, ith, offset_list[j])

                    if len(duration_list):
                        # if DBG_OPTIONS.PrintAction:
                        #     print(f'DBG duration_list_{i}={duration_list} changed')

                        for j in range(len(duration_list)):
                            tlid = an_sa_tlid_list[j]
                            ith = self.target_tl_id_list.index(tlid)
                            replaceTsoOutputInfoDuration(self.tso_output_info_dic, ith, duration_list[j])

            else:
                # self.action_mgmt.changePhaseArray(self.simulation_steps, i, actions[i])
                self.action_mgmt.changePhaseArray(self.simulation_steps, i, self.discrete_actions[i])


        # apply changed phase array and increase simulation steps
        if self.args.action in set(["offset", "gr", "gro"]):
            #--clculate how many steps to increase
            next_act, idx_of_next_act_sa = self.__getTimeToActInfo()
            next_act = next_act if next_act < self.end_step else self.end_step
            inc_step = next_act - self.simulation_steps

            #-- apply signal pahse, increase simulation step, and gather reward related info
            for i in range(inc_step):
                # 1. apply signal phase
                self.action_mgmt.applyCurrentTrafficSignalPhaseToEnv(self.simulation_steps)

                # 2. increase simulation step
                libsalt.simulationStep()
                self.simulation_steps += 1

                #3. gather reward related info
                if self.simulation_steps % self.reward_info_collection_cycle == 0:
                    # self.reward_mgmt.gatherRewardRelatedInfo(self.action_t, self.simulation_steps, self.reward_info_collection_cycle)
                    self.reward_mgmt.gatherRewardRelatedInfo(self.simulation_steps)

                # 4. gather visualization related info
                if self.args.mode == 'test':
                    appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                       self.discrete_actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                       self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

        elif self.args.action == "kc":  # keep or change
            idx_of_next_act_sa = list(range(self.agent_num))

            ## apply keep-change actions : first step
            current_phase_list = self.action_mgmt.applyKeepChangeActionFirstStep(self.simulation_steps, self.discrete_actions, self.tl_obj)

                # todo 반환값의 용도가 없다... 나중에 페이즈 정보 출력에 사용할 수 있을 지 모르겠다...
                # action 적용 전 현재 신호 페이즈 정보 저장

            ## increase simulation steps
            for i in range(3):  # todo should avoid using CONST 3
                libsalt.simulationStep()
                self.simulation_steps += 1

                # gather visualization related info
                if self.args.mode == 'test':
                    appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                       self.discrete_actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                       self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

            ## apply keep-change actions : second step
            next_phase_list = self.action_mgmt.applyKeepChangeActionSecondStep(self.simulation_steps, self.discrete_actions, self.tl_obj)

                # todo 반환값의 용도가 없다... 나중에 페이즈 정보 출력에 사용할 수 있을 지 모르겠다...
                # action 적용된 신호 페이즈 정보 저장

            ## increase simulation steps
            for i in range(self.action_t):
                libsalt.simulationStep()
                self.simulation_steps += 1

                # gather visualization related info
                if self.args.mode == 'test':
                    appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                       self.discrete_actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                       self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

        # for SAs to apply action next time (다음 번에 action을 적용할 SA들에 대해)
        #   1) calculate reward, 2) gather state info, 3) increase time to act
        for sa_i in idx_of_next_act_sa:
            said = self.sa_name_list[sa_i] # ...signal group name을  sa_i로 얻어온다.
            ##-- 1. calculate reward : only for target SA; do not gather reward for infer SA
            if said in self.target_sa_name_list:
                self.reward_mgmt.calculateReward(sa_i) # 보상 계산

            ##-- 2. gather state info
            self.observations[sa_i] = self.__getState(self.sa_obj[said], self.tl_obj)

            ##-- 3. increase time to act
            self.time_to_act_list[sa_i] = self.__getNextTimeToAct(self.simulation_steps, self.sa_cycle_list[sa_i], self.control_cycle)

            if DBG_OPTIONS.PrintStep:
                print("self.discrete_actions={}".format(self.discrete_actions))
                print("step={} sa_i={}  said={} tl_name={} discrete_actions={} rewards={}".
                                format(self.simulation_steps, sa_i, said,
                                self.sa_obj[said]['crossName_list'],
                                np.round(self.discrete_actions[sa_i], 3),
                                np.round(self.reward_mgmt.sa_rewards[sa_i], 2)))

        self.idx_of_act_sa = idx_of_next_act_sa

        if self.simulation_steps >= self.end_step:
            self.done = True
            print("self.done step {}".format(self.simulation_steps))
            libsalt.close()

        info = {}

        return self.observations, self.reward_mgmt.sa_rewards, self.done, info





    def stepOrg(self, actions):
        '''
        apply actions
        and gather rewards & observations

        :param actions:
        :return:
        '''
        self.done = False

        # change phase array by applying action
        #-- action에 따라 신호 페이즈 집합(self.action_mgmt.apply_phase_array_list)을 변경한다.
        #   지난 번 step에서 새로운 action을 적용할 시간이 도래한 것들에 대해 action 적용하여 변경
        for i in self.idx_of_act_sa:
            if DBG_OPTIONS.RichActionOutput:
                offset_list, duration_list = self.action_mgmt.changePhaseArray(self.simulation_steps, i, actions[i])

                if self.args.mode=='test':
                    sa_name = self.sa_name_list[i]
                    an_sa_obj = self.sa_obj[sa_name]
                    an_sa_tlid_list = an_sa_obj['tlid_list']

                    if len(offset_list):
                        # if DBG_OPTIONS.PrintAction:
                        #     print(f'DBG offset_list_{i}={offset_list} changed')

                        for j in range(len(offset_list)):
                            tlid = an_sa_tlid_list[j]
                            ith = self.target_tl_id_list.index(tlid)
                            assert ith < len(self.tso_output_info_dic["offset"]), print(f'ith={ith} len(self.tso_output_info_dic["offset"])={len(self.tso_output_info_dic["offset"])}')
                            replaceTsoOutputInfoOffset(self.tso_output_info_dic, ith, offset_list[j])

                    if len(duration_list):
                        # if DBG_OPTIONS.PrintAction:
                        #     print(f'DBG duration_list_{i}={duration_list} changed')

                        for j in range(len(duration_list)):
                            tlid = an_sa_tlid_list[j]
                            ith = self.target_tl_id_list.index(tlid)
                            replaceTsoOutputInfoDuration(self.tso_output_info_dic, ith, duration_list[j])

            else:
                self.action_mgmt.changePhaseArray(self.simulation_steps, i, actions[i])

        # apply changed phase array and increase simulation steps
        if self.args.action in set(["offset", "gr", "gro"]):
            #--clculate how many steps to increase
            next_act, idx_of_next_act_sa = self.__getTimeToActInfo()
            next_act = next_act if next_act < self.end_step else self.end_step
            inc_step = next_act - self.simulation_steps

            #-- apply signal pahse, increase simulation step, and gather reward related info
            for i in range(inc_step):
                # 1. apply signal phase
                self.action_mgmt.applyCurrentTrafficSignalPhaseToEnv(self.simulation_steps)

                # 2. increase simulation step
                libsalt.simulationStep()
                self.simulation_steps += 1

                #3. gather reward related info
                if self.simulation_steps % self.reward_info_collection_cycle == 0:
                    # self.reward_mgmt.gatherRewardRelatedInfo(self.action_t, self.simulation_steps, self.reward_info_collection_cycle)
                    self.reward_mgmt.gatherRewardRelatedInfo(self.simulation_steps)

                # 4. gather visualization related info
                if self.args.mode == 'test':
                    appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                       actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                       self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

        elif self.args.action == "kc":  # keep or change
            idx_of_next_act_sa = list(range(self.agent_num))

            ## apply keep-change actions : first step
            current_phase_list = self.action_mgmt.applyKeepChangeActionFirstStep(self.simulation_steps, actions, self.tl_obj)
                # todo 반환값의 용도가 없다... 나중에 페이즈 정보 출력에 사용할 수 있을 지 모르겠다...
                # action 적용 전 현재 신호 페이즈 정보 저장

            ## increase simulation steps
            for i in range(3):  # todo should avoid using CONST 3
                libsalt.simulationStep()
                self.simulation_steps += 1

                # gather visualization related info
                if self.args.mode == 'test':
                    appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                       actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                       self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

            ## apply keep-change actions : second step
            next_phase_list = self.action_mgmt.applyKeepChangeActionSecondStep(self.simulation_steps, actions, self.tl_obj)
                # todo 반환값의 용도가 없다... 나중에 페이즈 정보 출력에 사용할 수 있을 지 모르겠다...
                # action 적용된 신호 페이즈 정보 저장

            ## increase simulation steps
            for i in range(self.action_t):
                libsalt.simulationStep()
                self.simulation_steps += 1

                # gather visualization related info
                if self.args.mode == 'test':
                    appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                       actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                       self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

        # for SAs to apply action next time (다음 번에 action을 적용할 SA들에 대해)
        #   1) calculate reward, 2) gather state info, 3) increase time to act
        for sa_i in idx_of_next_act_sa:
            said = self.sa_name_list[sa_i] # ...signal group name을  sa_i로 얻어온다.
            ##-- 1. calculate reward : only for target SA; do not gather reward for infer SA
            if said in self.target_sa_name_list:
                self.reward_mgmt.calculateReward(sa_i) # 보상 계산

            ##-- 2. gather state info
            self.observations[sa_i] = self.__getState(self.sa_obj[said], self.tl_obj)

            ##-- 3. increase time to act
            self.time_to_act_list[sa_i] = self.__getNextTimeToAct(self.simulation_steps, self.sa_cycle_list[sa_i], self.control_cycle)

            if DBG_OPTIONS.PrintStep:
                print("actions={}".format(actions))
                print("step={} sa_i={}  said={} tl_name={} actions={} rewards={}".format(self.simulation_steps, sa_i, said,
                                                                                         self.sa_obj[said]['crossName_list'],
                                                                                         np.round(actions[sa_i], 3),
                                                                                         np.round(self.reward_mgmt.sa_rewards[sa_i], 2)))

        self.idx_of_act_sa = idx_of_next_act_sa

        if self.simulation_steps >= self.end_step:
            self.done = True
            print("self.done step {}".format(self.simulation_steps))
            libsalt.close()

        info = {}

        return self.observations, self.reward_mgmt.sa_rewards, self.done, info




    def reset(self):
        '''
        initialize simulation
        :return:
        '''
        libsalt.start(self.salt_scenario, self.args.output_home)
        libsalt.setCurrentStep(self.start_step)
        self.simulation_steps = libsalt.getCurrentStep()

        if self.args.mode == 'test':
            for k in self.tso_output_info_dic:
                self.tso_output_info_dic[k].clear()

            for tlid in self.target_tl_id_list:
                avg_speed, avg_tt, sum_passed, sum_travel_time = gatherTsoOutputInfo(tlid, self.tl_obj, num_hop=0)

                if DBG_OPTIONS.RichActionOutput:
                    #todo should consider the possibility that TOD can be changed
                    offset = self.tl_obj[tlid]['offset']
                    duration = self.tl_obj[tlid]['duration']

                    if DBG_OPTIONS.PrintAction:
                        cross_name = self.tl_obj[tlid]['crossName']
                        green_idx = self.tl_obj[tlid]['green_idx']
                        print(
                            f'cross_name={cross_name} offset={offset} duration={duration} green_idx={green_idx}  green_idx[0]={green_idx[0]}')

                    appendTsoOutputInfoSignal(self.tso_output_info_dic, offset, duration)
                self.tso_output_info_dic = appendTsoOutputInfo(self.tso_output_info_dic, avg_speed, avg_tt, sum_passed, sum_travel_time)

        #-- warming up
        ##--- make dummy actions to write output file
        self.discrete_actions.clear()

        for i in range(len(self.sa_name_list)):
            target_sa = self.sa_name_list[i]
            action_space = self.sa_obj[target_sa]['action_space']
            action_size = action_space.shape[0]
            self.discrete_actions.append(list(0 for _ in range(action_size)))
                    # zero because the offset of the fixed signal is used as it is

            if 1:
                print(f"Reset discrete_actions={self.discrete_actions}")
                print(f"Reset sa={target_sa}  action_space={action_space}  action_space.shape[0]={action_space.shape[0]}")


        ##--- increase simulation steps
        for _ in range(self.warming_up_time):
            libsalt.simulationStep()
            self.simulation_steps += 1

            # gather visualization related info
            if self.args.mode == 'test':
                appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                   self.discrete_actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                   self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

        self.simulation_steps = libsalt.getCurrentStep()

        #-- initialize : reward, time_to_act_list, observation
        self.reward_mgmt.reset()

        for i in range(len(self.sa_name_list)):
            self.time_to_act_list[i] = self.__getNextTimeToAct(self.simulation_steps, self.sa_cycle_list[i],
                                                               self.control_cycle)

        self.observations = list([] for i in range(self.agent_num))  # [ [], ...,[]]

        #
        # action 을 적용해야 하는 곳까지 시뮬레이션을 수행한다.
        # 이때, 보상 관연 정보를 수집한다. 또한, agent.act() 의 입력이 되는 상태 정보를 수집한다.
        # performs simulation until the action needs to be applied
        idx_of_next_act_sa = []
        if self.args.action in set(["offset", "gr", "gro"]):
            #--- 1. find the time when the action should be applied through inference
            #       and get index of SA to determine action through model inference
            next_act, idx_of_next_act_sa = self.__getTimeToActInfo()
            inc_step = next_act - self.simulation_steps
            self.idx_of_act_sa = idx_of_next_act_sa

            #--- 2. increase simulation step, gather reward related info
            for i in range(inc_step):
                libsalt.simulationStep()
                self.simulation_steps += 1

                # gather visualization related info
                if self.args.mode == 'test':
                    appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                       self.discrete_actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                       self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

                if self.simulation_steps % self.reward_info_collection_cycle == 0:
                    # self.reward_mgmt.gatherRewardRelatedInfo(self.action_t, self.simulation_steps, self.reward_info_collection_cycle)
                    self.reward_mgmt.gatherRewardRelatedInfo(self.simulation_steps)

        elif self.args.action == "kc":
            idx_of_next_act_sa = list(range(self.agent_num))

        assert  len(idx_of_next_act_sa) != 0, f"internal error : action ({self.args.action}) is not cared"

        #--- 3. gather state info and get next action-time
        for i in idx_of_next_act_sa:
            self.observations[i] = self.__getState(self.sa_obj[self.sa_name_list[i]], self.tl_obj)
            self.time_to_act_list[i] = self.__getNextTimeToAct(self.simulation_steps, self.sa_cycle_list[i], self.control_cycle)

        return self.observations

    def resetOrg(self):
        '''
        initialize simulation
        :return:
        '''
        libsalt.start(self.salt_scenario, self.args.output_home)
        libsalt.setCurrentStep(self.start_step)
        self.simulation_steps = libsalt.getCurrentStep()

        if self.args.mode == 'test':
            for k in self.tso_output_info_dic:
                self.tso_output_info_dic[k].clear()

            for tlid in self.target_tl_id_list:
                avg_speed, avg_tt, sum_passed, sum_travel_time = gatherTsoOutputInfo(tlid, self.tl_obj, num_hop=0)

                if DBG_OPTIONS.RichActionOutput:
                    # todo should consider the possibility that TOD can be changed
                    offset = self.tl_obj[tlid]['offset']
                    duration = self.tl_obj[tlid]['duration']

                    if DBG_OPTIONS.PrintAction:
                        cross_name = self.tl_obj[tlid]['crossName']
                        green_idx = self.tl_obj[tlid]['green_idx']
                        print(
                            f'cross_name={cross_name} offset={offset} duration={duration} green_idx={green_idx}  green_idx[0]={green_idx[0]}')

                    appendTsoOutputInfoSignal(self.tso_output_info_dic, offset, duration)
                self.tso_output_info_dic = appendTsoOutputInfo(self.tso_output_info_dic, avg_speed, avg_tt, sum_passed,
                                                               sum_travel_time)

        # -- warming up
        ##--- make dummy actions to write output file
        actions = []

        for i in range(len(self.sa_name_list)):
            target_sa = self.sa_name_list[i]
            action_space = self.sa_obj[target_sa]['action_space']
            action_size = action_space.shape[0]
            actions.append(list(0 for _ in range(action_size)))

            if 1:
                print(f"Reset action={actions}")
                print(
                    f"Reset sa={target_sa}  action_space={action_space}  action_space.shape[0]={action_space.shape[0]}")

        ##--- increase simulation steps
        for _ in range(self.warming_up_time):
            libsalt.simulationStep()
            self.simulation_steps += 1

            # gather visualization related info
            if self.args.mode == 'test':
                appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                   actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                   self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

        self.simulation_steps = libsalt.getCurrentStep()

        # -- initialize : reward, time_to_act_list, observation
        self.reward_mgmt.reset()

        for i in range(len(self.sa_name_list)):
            self.time_to_act_list[i] = self.__getNextTimeToAct(self.simulation_steps, self.sa_cycle_list[i],
                                                               self.control_cycle)

        self.observations = list([] for i in range(self.agent_num))  # [ [], ...,[]]

        #
        # action 을 적용해야 하는 곳까지 시뮬레이션을 수행한다.
        # 이때, 보상 관연 정보를 수집한다. 또한, agent.act() 의 입력이 되는 상태 정보를 수집한다.
        # performs simulation until the action needs to be applied
        idx_of_next_act_sa = []
        if self.args.action in set(["offset", "gr", "gro"]):
            # --- 1. find the time when the action should be applied through inference
            #       and get index of SA to determine action through model inference
            next_act, idx_of_next_act_sa = self.__getTimeToActInfo()
            inc_step = next_act - self.simulation_steps
            self.idx_of_act_sa = idx_of_next_act_sa

            # --- 2. increase simulation step, gather reward related info
            for i in range(inc_step):
                libsalt.simulationStep()
                self.simulation_steps += 1

                # gather visualization related info
                if self.args.mode == 'test':
                    appendPhaseRewards(self.fn_rl_phase_reward_output, self.simulation_steps,
                                       actions, self.reward_mgmt, self.sa_obj, self.sa_name_list,
                                       self.tl_obj, self.target_tl_id_list, self.tso_output_info_dic)

                if self.simulation_steps % self.reward_info_collection_cycle == 0:
                    # self.reward_mgmt.gatherRewardRelatedInfo(self.action_t, self.simulation_steps, self.reward_info_collection_cycle)
                    self.reward_mgmt.gatherRewardRelatedInfo(self.simulation_steps)

        elif self.args.action == "kc":
            idx_of_next_act_sa = list(range(self.agent_num))

        assert len(idx_of_next_act_sa) != 0, f"internal error : action ({self.args.action}) is not cared"

        # --- 3. gather state info and get next action-time
        for i in idx_of_next_act_sa:
            self.observations[i] = self.__getState(self.sa_obj[self.sa_name_list[i]], self.tl_obj)
            self.time_to_act_list[i] = self.__getNextTimeToAct(self.simulation_steps, self.sa_cycle_list[i],
                                                               self.control_cycle)

        return self.observations

    def render(self, mode='human'):
        pass
        # print(self.reward)


    def close(self):
        libsalt.close()
        print('close')


    def isTrainTarget(self, sa_name):
        '''
        check if given SA is in target to be trained

        :param sa_name:
        :return:  True or False
        '''

        return sa_name in self.target_sa_name_list