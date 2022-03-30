import gym
import shutil
import uuid
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import os
import numpy as np
from xml.etree.ElementTree import parse
import collections
import math
import copy
from config import TRAIN_CONFIG
# print(TRAIN_CONFIG)
sys.path.append(TRAIN_CONFIG['libsalt_dir'])

import libsalt

state_weight = 1
reward_weight = 1
addTime = 1

sim_period = 30

from config import TRAIN_CONFIG

IS_DOCKERIZE = TRAIN_CONFIG['IS_DOCKERIZE']

from env.get_objs import get_objs

import json
import platform

def getScenarioRelatedFilePath(args):
    abs_scenario_file_path = '{}/{}'.format(os.getcwd(), args.scenario_file_path)

    input_file_path = os.path.dirname(abs_scenario_file_path)
    if platform.system() == 'Windows':  # one of { Windows, Linux , Darwin }
        dir_delimiter = "\\"
    else:
        dir_delimiter = "/"

    with open(abs_scenario_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        node_file = json_data["scenario"]["input"]["node"]
        edge_file = json_data["scenario"]["input"]["link"]
        tss_file = json_data["scenario"]["input"]["trafficLightSystem"]

    node_file_path = input_file_path + dir_delimiter + node_file
    edge_file_path = input_file_path + dir_delimiter + edge_file
    tss_file_path = input_file_path + dir_delimiter + tss_file

    return abs_scenario_file_path, node_file_path, edge_file_path, tss_file_path

def getScenarioRelatedBeginEndTime(scenario_file_path):
    abs_scenario_file_path = '{}/{}'.format(os.getcwd(), scenario_file_path)

    with open(abs_scenario_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        begin_time = json_data["scenario"]["time"]["begin"]
        end_time = json_data["scenario"]["time"]["end"]

    return begin_time, end_time

class SALT_SAPPO_offset_EA(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.state_weight = state_weight
        self.reward_weight = reward_weight
        self.addTime = addTime
        self.reward_func = args.reward_func
        self.actionT = args.action_t
        self.args = args
        self.cp = args.cp
        self.printOut = args.printOut
        self.sim_period = sim_period

        if IS_DOCKERIZE:
            scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)

            # self.startStep = args.trainStartTime if args.trainStartTime > scenario_begin else scenario_begin
            # self.endStep = args.trainEndTime if args.trainEndTime < scenario_end else scenario_end
            self.startStep = args.start_time if args.start_time > scenario_begin else scenario_begin
            self.endStep = args.end_time if args.end_time < scenario_end else scenario_end
        else:
            self.startStep = args.trainStartTime
            self.endStep = args.trainEndTime

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.uid = str(uuid.uuid4())

        if IS_DOCKERIZE:
            abs_scenario_file_path = '{}/{}'.format(os.getcwd(), args.scenario_file_path)
            self.src_dir = os.path.dirname(abs_scenario_file_path)
            self.dest_dir = os.path.split(self.src_dir)[0]
            self.dest_dir = '{}/data/{}/'.format(self.dest_dir, self.uid)
            os.makedirs(self.dest_dir, exist_ok=True)
        else:
            if self.args.map == 'dj':
                self.src_dir = os.getcwd() + "/data/envs/salt/dj_all"
            if self.args.map == 'doan':
                self.src_dir = os.getcwd() + "/data/envs/salt/doan"
            self.dest_dir = os.getcwd() + "/data/envs/salt/data/" + self.uid + "/"
            os.mkdir(self.dest_dir)

        src_files = os.listdir(self.src_dir)
        for file_name in src_files:
            full_file_name = os.path.join(self.src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, self.dest_dir)

        if IS_DOCKERIZE:
            scenario_file_name = args.scenario_file_path.split('/')[-1]
            self.salt_scenario = "{}/{}".format(self.dest_dir, scenario_file_name)
            _, _, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args)
            tree = parse(tss_file_path)
        else:
            # self.salt_scenario = self.dest_dir + 'doan_2021_actionT{}.scenario.json'.format(self.actionT)
            if args.mode == 'train':
                if self.args.map == 'dj':
                    self.salt_scenario = self.dest_dir + 'dj_all.scenario.json'
                if self.args.map == 'doan':
                    self.salt_scenario = self.dest_dir + 'doan_20211207.scenario.json'
            if args.mode == 'test':
                if self.args.map == 'dj':
                    self.salt_scenario = self.dest_dir + 'dj_all_test.scenario.json'
                if self.args.map == 'doan':
                    self.salt_scenario = self.dest_dir + 'doan_20211207_test.scenario.json'

            if self.args.map == 'dj':
                edge_file_path = "data/dj_all/edge.xml"
                tss_file_path = "data/envs/salt/dj_all/tss.xml"
                tree = parse(os.getcwd() + '/data/envs/salt/dj_all/tss.xml')
            if self.args.map == 'doan':
                edge_file_path = "magic/doan_20211207.edg.xml"
                tss_file_path = "magic/doan_20211207.tss.xml"
                tree = parse(os.getcwd() + '/data/envs/salt/doan/doan_20211207.tss.xml')


        root = tree.getroot()

        trafficSignal = root.findall("trafficSignal")

        self.phase_numbers = []
        i=0

        self.targetList_input = args.target_TL.split(',')

        self.targetList_input2 = []

        for tl_i in self.targetList_input:
            self.targetList_input2.append(tl_i)                         ## ex. SA 101
            self.targetList_input2.append(tl_i.split(" ")[1])           ## ex.101
            self.targetList_input2.append(tl_i.replace(" ", ""))        ## ex. SA101

        self.target_tl_obj, self.sa_obj, _lane_len = get_objs(args, trafficSignal, self.targetList_input2, edge_file_path, self.salt_scenario, self.startStep)

        self.target_tl_id_list = list(self.target_tl_obj.keys())

        self.agent_num = len(self.sa_obj)

        self.control_cycle = args.controlcycle

        print('target tl obj {}'.format(self.target_tl_obj))
        print('target tl id list {}'.format(self.target_tl_id_list))
        print('number of target tl {}'.format(len(self.target_tl_id_list)))

        self.max_lane_length = np.max(_lane_len)
        print(self.target_tl_obj)
        print(np.max(_lane_len))
        self.observations = []
        self.lane_passed = []

        self.phase_arr = []
        self.action_mask = []

        for target in self.sa_obj:
            self.observations.append([0] * self.sa_obj[target]['state_space'])
            # print(self.sa_obj[target]['action_min'], self.sa_obj[target]['action_max'])
            self.sa_obj[target]['action_space'] = spaces.Box(low=np.array(self.sa_obj[target]['action_min']), high=np.array(self.sa_obj[target]['action_max']), dtype=np.int32)

            self.lane_passed.append([])
            self.phase_arr.append([])
            self.action_mask = np.append(self.action_mask, 0)


        # print("self.lane_passed", self.lane_passed)
        # print(self.observations)
        # print("self.action_keep_time", self.action_keep_time)
        self.rewards = np.zeros(self.agent_num)

        self.before_action = []
        self.real_action = []
        self.virtual_action = []
        self.virtual_action_return = []
        for target_sa in self.sa_obj:
            self.before_action.append([0] * self.sa_obj[target_sa]['action_space'].shape[0])
            self.real_action.append([0] * self.sa_obj[target_sa]['action_space'].shape[0])
            self.virtual_action.append([0] * self.sa_obj[target_sa]['action_space'].shape[0])
            self.virtual_action_return.append([0] * self.sa_obj[target_sa]['action_space'].shape[0])
        print("before action", self.before_action)

        # print('{} traci closed\n'.format(self.uid))

        self.simulationSteps = 0

    def step(self, actions):
        self.done = False
        # print('step')

        currentStep = libsalt.getCurrentStep()
        self.simulationSteps = currentStep

        sa_i = 0
        for sa in self.sa_obj:
            tlid_list = self.sa_obj[sa]['tlid_list']
            sa_cycle = self.sa_obj[sa]['cycle_list'][0]

            if self.simulationSteps % (self.sa_obj[sa]['cycle_list'][0]*self.control_cycle) == 0:
                self.real_action = copy.deepcopy(actions)
                self.virtual_action = copy.deepcopy(actions)
                self.virtual_action_return = copy.deepcopy(actions)
                for i in range(len(self.virtual_action[sa_i])):
                    self.virtual_action_return[sa_i][i] = self.virtual_action[sa_i][i]*2/sa_cycle
                # print(tlid_list)
                self.phase_arr[sa_i] = []
                for tlid_i in range(len(tlid_list)):
                    currDur = self.sa_obj[sa]['duration_list'][tlid_i]
                    phase_arr = []
                    for i in range(len(currDur)):
                        phase_arr = np.append(phase_arr, np.ones(currDur[i]) * i)
                    self.phase_arr[sa_i].append(np.roll(phase_arr, self.sa_obj[sa]['offset_list'][tlid_i] + actions[sa_i][tlid_i]))
            else:
                self.virtual_action[sa_i] = np.array(self.virtual_action[sa_i]) + self.sim_period
                for i in range(len(self.virtual_action[sa_i])):
                    if self.virtual_action[sa_i][i] > sa_cycle/2:
                        self.virtual_action[sa_i][i] = -(sa_cycle-self.virtual_action[sa_i][i])
                    self.virtual_action_return[sa_i][i] = self.virtual_action[sa_i][i]*2/sa_cycle

            for _ in range(self.sim_period):
                for tlid_i in range(len(tlid_list)):
                    t_phase = int(self.phase_arr[sa_i][tlid_i][self.simulationSteps % sa_cycle])
                    scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid_list[tlid_i])
                    libsalt.trafficsignal.changeTLSPhase(currentStep, tlid_list[tlid_i], scheduleID, t_phase)
                libsalt.simulationStep()
                currentStep = libsalt.getCurrentStep()
                self.simulationSteps = currentStep

            self.observations[sa_i] = self.get_state(sa)

            link_list_0 = self.sa_obj[sa]['in_edge_list_0']
            link_list_1 = self.sa_obj[sa]['in_edge_list_1']
            lane_list_0 = self.sa_obj[sa]['in_lane_list_0']
            lane_list_1 = self.sa_obj[sa]['in_lane_list_1']

            if self.reward_func == 'pn':
                for l in link_list_0:
                    self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumPassed(l))
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumPassed(l) * reward_weight)
            if self.reward_func == 'wt':
                for l in link_list_0:
                    self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingTime(l) / self.actionT)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingTime(l) / self.actionT * reward_weight)
            if self.reward_func == 'wt_max':
                for l in link_list_0:
                    self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingTime(l) / self.actionT)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingTime(l) / self.actionT * reward_weight)
            if self.reward_func in ['wq', 'wq_median', 'wq_min', 'wq_max']:
                for l in link_list_0:
                    # print("sum([l in x for x in lane_list_0])", sum([l in x for x in lane_list_0]))
                    self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingQLength(l) * sum([l in x for x in lane_list_0]))
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getAverageWaitingQLength(l) * reward_weight)
            if self.reward_func == 'wt_SBV':
                for l in link_list_0:
                    self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps) / 1000)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps) / 1000 * reward_weight)
            if self.reward_func == 'wt_SBV_max':
                for l in link_list_0:
                    self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps) / 1000)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps) / 1000 * reward_weight)
            if self.reward_func == 'wt_ABV':
                for l in link_list_0:
                    self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulationSteps) / 1000)
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulationSteps) / 1000 * reward_weight)
            if self.reward_func == 'tt':
                for l in link_list_0:
                    # self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumTravelTime(l))
                    self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumTravelTime(l) / (len(link_list_0) * self.sim_period))
                # for l in link_list_1:
                #     self.lane_passed[sa_i] = np.append(self.lane_passed[sa_i], libsalt.link.getSumTravelTime(l) / 1000 * reward_weight)


            if self.printOut:
                print("step {} tl_name {} real_actions {} virtual_actions {} rewards {}".format(self.simulationSteps,
                                                                                          self.sa_obj[sa]['crossName_list'],
                                                                                          np.round(self.real_action[sa_i], 3),
                                                                                          np.round(self.virtual_action[sa_i], 3),
                                                                                          np.round(self.rewards[sa_i], 2)))
            sa_i += 1

        if self.simulationSteps >= self.endStep:
            self.done = True
            print("self.done step {}".format(self.simulationSteps))
            libsalt.close()
            sa_i = 0
            for sa in self.sa_obj:
                if self.reward_func == 'pn':
                    self.rewards[sa_i] = np.sum(self.lane_passed[sa_i])
                if self.reward_func == 'wt':
                    self.rewards[sa_i] = -np.sum(self.lane_passed[sa_i])
                if self.reward_func == 'wt_max':
                    self.rewards[sa_i] = -np.max(self.lane_passed[sa_i])
                if self.reward_func == 'wq':
                    self.rewards[sa_i] = -np.sum(self.lane_passed[sa_i])
                if self.reward_func == 'wq_median':
                    if len(self.lane_passed[sa_i]) == 0:
                        self.rewards[sa_i] = 0
                    else:
                        self.lane_passed[sa_i][self.lane_passed[sa_i] == 0] = np.nan
                        if np.isnan(np.nanmedian(self.lane_passed[sa_i])):
                            self.rewards[sa_i] = 0
                        else:
                            self.rewards[sa_i] = -np.nanmedian(self.lane_passed[sa_i])
                if self.reward_func == 'wq_min':
                    self.rewards[sa_i] = -np.min(self.lane_passed[sa_i])
                if self.reward_func == 'wq_max':
                    if len(self.lane_passed[sa_i]) == 0:
                        self.rewards[sa_i] = 0
                    else:
                        self.rewards[sa_i] = -np.max(self.lane_passed[sa_i])
                if self.reward_func == 'wt_SBV':
                    self.rewards[sa_i] = -np.sum(self.lane_passed[sa_i])
                if self.reward_func == 'wt_SBV_max':
                    self.rewards[sa_i] = -np.mean(self.lane_passed[sa_i])
                if self.reward_func == 'wt_ABV':
                    self.rewards[sa_i] = -np.mean(self.lane_passed[sa_i])
                if self.reward_func == 'tt':
                    # self.lane_passed[sa_i] = self.lane_passed[sa_i] + np.finfo(float).eps
                    # self.lane_passed[sa_i][self.lane_passed[sa_i]==0] = np.nan
                    # self.rewards[sa_i] = -np.nanmean(self.lane_passed[sa_i] / np.nanmax(self.lane_passed[sa_i]))
                    self.rewards[sa_i] = -np.sum(self.lane_passed[sa_i])
                self.lane_passed[sa_i] = []
            sa_i += 1


        info = {}
        # print(self.before_action, actions)
        self.before_action = actions.copy()

        return self.observations, self.rewards, self.done, info, self.virtual_action_return

    def reset(self):
        print("reset")
        libsalt.start(self.salt_scenario)
        libsalt.setCurrentStep(self.startStep)

        self.simulationSteps = libsalt.getCurrentStep()

        observations = []
        self.lane_passed = []
        self.phase_arr = []
        sa_i=0
        self.action_mask = []
        self.rewards = np.zeros(self.agent_num)

        self.before_action = []
        self.real_action = []
        self.virtual_action = []
        for target_sa in self.sa_obj:
            self.before_action.append([0] * self.sa_obj[target_sa]['action_space'].shape[0])
            self.real_action.append([0] * self.sa_obj[target_sa]['action_space'].shape[0])
            self.virtual_action.append([0] * self.sa_obj[target_sa]['action_space'].shape[0])

        for said in self.sa_obj:
            # print(f"said{said}", self.get_state(said))
            self.lane_passed.append([])
            self.phase_arr.append([])
            self.phase_arr[sa_i] = []
            tlid_list = self.sa_obj[said]['tlid_list']
            tlid_i = 0
            for tlid in tlid_list:
                currDur = self.sa_obj[said]['duration_list'][tlid_i]

                phase_arr = []
                for i in range(len(currDur)):
                    phase_arr = np.append(phase_arr, np.ones(currDur[i]) * i)

                self.phase_arr[sa_i].append(phase_arr)
                tlid_i += 1
            self.action_mask = np.append(self.action_mask, 0)
            observations.append(self.get_state(said))

            sa_i += 1

        return observations

    def get_state(self, said):
        # print(said)
        obs = []
        densityMatrix = []
        passedMatrix = []
        vddMatrix = []
        tlMatrix = []

        for tlid in self.sa_obj[said]['tlid_list']:
            link_list = self.target_tl_obj[tlid]['in_lane_list']
            # print(link_list)
            link_list_0 = self.target_tl_obj[tlid]['in_lane_list_0']
            link_list_1 = self.target_tl_obj[tlid]['in_lane_list_1']

            # tl_s = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tsid).myPhaseVector[0][1]
            #print("tl_s {}".format(tl_s))

            for link in link_list_0:
                if self.args.state == 'd':
                    densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link))
                if self.args.state == 'v':
                    passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link))
                if self.args.state == 'vd':
                    densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link))
                    passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link))
                if self.args.state == 'vdd':
                    vddMatrix = np.append(vddMatrix, libsalt.lane.getNumVehPassed(link)/(libsalt.lane.getAverageDensity(link)+sys.float_info.epsilon))
            # for link in link_list_1:
            #     if self.args.state == 'd':
            #         densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link) * self.state_weight)
            #     if self.args.state == 'v':
            #         passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link) * self.state_weight)
            #     if self.args.state == 'vd':
            #         densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link) * self.state_weight)
            #         passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link) * self.state_weight)
            #     if self.args.state == 'vdd':
            #         vddMatrix = np.append(vddMatrix, libsalt.lane.getNumVehPassed(link)/(libsalt.lane.getAverageDensity(link)+sys.float_info.epsilon) * self.state_weight)

            tlMatrix = np.append(tlMatrix, libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid))
            #print(lightMatrix)
            # obs = np.append(densityMatrix, passedMatrix)
            if self.args.state == 'd':
                obs = np.append(densityMatrix, tlMatrix)
            if self.args.state == 'v':
                obs = np.append(passedMatrix, tlMatrix)
            if self.args.state == 'vd':
                obs = np.append(densityMatrix, passedMatrix)
                obs = np.append(obs, tlMatrix)
            if self.args.state == 'vdd':
                obs = np.append(vddMatrix, tlMatrix)
        # print(obs)

        # if self.args.method=='sappo':
        obs = obs + np.finfo(float).eps
        # print(obs)
        obs = obs/np.max(obs)
        # print(obs)
        # print(densityMatrix)
        # print(passedMatrix)
        # print(f"get_state obs {obs} obslen {len(obs)}")
        # print(obs)
        return obs


    def render(self, mode='human'):
        pass
        # print(self.reward)

    def close(self):
        libsalt.close()
        print('close')
