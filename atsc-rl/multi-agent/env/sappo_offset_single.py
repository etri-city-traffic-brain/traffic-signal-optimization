import gym
import shutil
import uuid
from gym import error, spaces, utils
from gym.utils import seeding
import sys
import os
import numpy as np
from xml.etree.ElementTree import parse

from config import TRAIN_CONFIG
sys.path.append(TRAIN_CONFIG['libsalt_dir'])

import libsalt

state_weight = 1
reward_weight = 1

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

class SALT_SAPPO_offset_single(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.args = args
        self.state_weight = state_weight
        self.reward_weight = reward_weight
        self.reward_func = args.reward_func
        self.actionT = args.action_t
        self.cp = args.cp
        self.printOut = args.printOut
        self.sim_period = sim_period
        self.warmupTime = args.warmupTime
        self.control_cycle = args.controlcycle

        scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
        self.startStep = args.start_time if args.start_time > scenario_begin else scenario_begin
        self.endStep = args.end_time if args.end_time < scenario_end else scenario_end

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.uid = str(uuid.uuid4())

        abs_scenario_file_path = '{}/{}'.format(os.getcwd(), args.scenario_file_path)
        self.src_dir = os.path.dirname(abs_scenario_file_path)
        self.dest_dir = os.path.split(self.src_dir)[0]
        self.dest_dir = '{}/data/{}/'.format(self.dest_dir, self.uid)
        os.makedirs(self.dest_dir, exist_ok=True)

        src_files = os.listdir(self.src_dir)
        for file_name in src_files:
            full_file_name = os.path.join(self.src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, self.dest_dir)

        scenario_file_name = args.scenario_file_path.split('/')[-1]
        self.salt_scenario = "{}/{}".format(self.dest_dir, scenario_file_name)
        _, _, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args)
        tree = parse(tss_file_path)

        root = tree.getroot()

        trafficSignal = root.findall("trafficSignal")

        self.phase_numbers = []

        self.targetList_input = args.target_TL.split(',')
        self.targetList_input2 = []
        for tl_i in self.targetList_input:
            self.targetList_input2.append(tl_i)                         ## ex. SA 101
            self.targetList_input2.append(tl_i.split(" ")[1])           ## ex.101
            self.targetList_input2.append(tl_i.replace(" ", ""))        ## ex. SA101

        self.target_tl_obj, self.sa_obj, _lane_len = get_objs(args, trafficSignal, self.targetList_input2, edge_file_path, self.salt_scenario, self.startStep)
        self.target_tl_id_list = list(self.target_tl_obj.keys())

        self.agent_num = len(self.sa_obj)

        print('target tl obj {}'.format(self.target_tl_obj))
        print('target tl id list {}'.format(self.target_tl_id_list))
        print('number of target tl {}'.format(len(self.target_tl_id_list)))

        print(self.target_tl_obj)
        self.observations = []
        self.lane_passed = []

        self.phase_arr = []
        for target in self.sa_obj:
            self.observations.append([0] * self.sa_obj[target]['state_space'])
            self.sa_obj[target]['action_space'] = spaces.Box(low=np.array(self.sa_obj[target]['action_min']), high=np.array(self.sa_obj[target]['action_max']), dtype=np.int32)

            self.lane_passed.append([])
            self.phase_arr.append([])

        self.rewards = np.zeros(self.agent_num)

        self.before_action = []
        for target_sa in self.sa_obj:
            self.before_action.append([0] * self.sa_obj[target_sa]['action_space'].shape[0])
        print("before action", self.before_action)

        self.simulationSteps = 0

    def step(self, actions):
        self.done = False

        currentStep = libsalt.getCurrentStep()
        self.simulationSteps = currentStep

        sa_i = 0
        for sa in self.sa_obj:
            if self.simulationSteps % (self.sa_obj[sa]['cycle_list'][0]*self.control_cycle) == 0:
                tlid_list = self.sa_obj[sa]['tlid_list']
                sa_cycle = self.sa_obj[sa]['cycle_list'][0]

                _phase_sum = []
                _phase_list = []
                self.phase_arr[sa_i] = []
                for tlid_i in range(len(tlid_list)):
                    tlid = tlid_list[tlid_i]
                    currDur = self.sa_obj[sa]['duration_list'][tlid_i]
                    phase_arr = []
                    for i in range(len(currDur)):
                        phase_arr = np.append(phase_arr, np.ones(currDur[i]) * i)
                    self.phase_arr[sa_i].append(np.roll(phase_arr, self.sa_obj[sa]['offset_list'][tlid_i] + actions[sa_i][tlid_i]))
                    __phase_sum = np.sum([x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector])
                    _phase_sum.append(__phase_sum)
                    __phase_list = [x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector if x[0] > 5]
                    _phase_list.append(__phase_list)

                for _ in range(sa_cycle * self.control_cycle):
                    for tlid_i in range(len(tlid_list)):
                        tlid = tlid_list[tlid_i]
                        t_phase = int(self.phase_arr[sa_i][tlid_i][self.simulationSteps % sa_cycle])
                        scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
                        libsalt.trafficsignal.changeTLSPhase(currentStep, tlid, scheduleID, t_phase)
                    libsalt.simulationStep()
                    self.simulationSteps = libsalt.getCurrentStep()
                    currentStep = self.simulationSteps

                    if self.simulationSteps % self.sim_period == 0:
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
                        if self.reward_func in ['wq', 'wq_median', 'wq_min', 'wq_max', 'cwq']:
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
                        self.observations[sa_i] += self.get_state(sa)

                if self.simulationSteps > 0:
                    if self.reward_func == 'pn':
                        self.rewards[sa_i] = np.sum(self.lane_passed[sa_i])
                    if self.reward_func == 'wt':
                        self.rewards[sa_i] = -np.sum(self.lane_passed[sa_i])
                    if self.reward_func == 'wt_max':
                        self.rewards[sa_i] = -np.max(self.lane_passed[sa_i])
                    if self.reward_func == 'wq' or self.reward_func == 'cwq':
                        self.rewards[sa_i] = -np.sum(self.lane_passed[sa_i])/5000
                    if self.reward_func == 'wq_median':
                        if len(self.lane_passed[sa_i])==0:
                            self.rewards[sa_i] = 0
                        else:
                            self.lane_passed[sa_i][self.lane_passed[sa_i]==0] = np.nan
                            if np.isnan(np.nanmedian(self.lane_passed[sa_i])):
                                self.rewards[sa_i] = 0
                            else:
                                self.rewards[sa_i] = -np.nanmedian(self.lane_passed[sa_i])
                    if self.reward_func == 'wq_min':
                        self.rewards[sa_i] = -np.min(self.lane_passed[sa_i])
                    if self.reward_func == 'wq_max':
                        if len(self.lane_passed[sa_i])==0:
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
                    if self.reward_func != 'cwq':
                        self.lane_passed[sa_i] = []
                    self.observations[sa_i] = self.get_state(sa)
                else:
                    self.rewards[sa_i] = 0
                    # self.rewards[sa_i] += penalty
                    # self.lane_passed[sa_i] = []

                if self.printOut:
                    print("step {} tl_name {} actions {} rewards {}".format(self.simulationSteps,
                                                                                              self.sa_obj[sa]['crossName_list'],
                                                                                              np.round(actions[sa_i], 3),
                                                                                              np.round(self.rewards[sa_i], 2)))
            else:
                libsalt.simulationStep()
                self.simulationSteps = libsalt.getCurrentStep()
                currentStep = self.simulationSteps

            sa_i += 1

        if self.simulationSteps >= self.endStep:
            self.done = True
            print("self.done step {}".format(self.simulationSteps))
            libsalt.close()

        info = {}
        self.before_action = actions.copy()

        return self.observations, self.rewards, self.done, info

    def reset(self):
        print("reset")
        libsalt.start(self.salt_scenario)
        libsalt.setCurrentStep(self.startStep)

        self.simulationSteps = libsalt.getCurrentStep()

        for _ in range(self.warmupTime):
            libsalt.simulationStep()
        self.simulationSteps = libsalt.getCurrentStep()

        for said in self.sa_obj:
            while(self.simulationSteps % (self.sa_obj[said]['cycle_list'][0]*self.control_cycle)!=0):
                libsalt.simulationStep()
                self.simulationSteps = libsalt.getCurrentStep()

        print(f"{self.simulationSteps} start")

        observations = []
        self.lane_passed = []
        self.phase_arr = []
        sa_i=0
        self.rewards = np.zeros(self.agent_num)

        for said in self.sa_obj:
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
            observations.append(self.get_state(said))

            sa_i += 1

        return observations

    def get_state(self, said):
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

        obs = obs + np.finfo(float).eps
        obs = obs/np.max(obs)
        return obs


    def render(self, mode='human'):
        pass

    def close(self):
        libsalt.close()
        print('close')
