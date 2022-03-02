##### test 버전 만들 때 주의 사항 ####
#### 1. test 시나리오 파일로 변경
#### 2. phase_txt 파일 만들기 위한 init 함수, step 함수 변경

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

from config import TRAIN_CONFIG
# print(TRAIN_CONFIG)
sys.path.append(TRAIN_CONFIG['libsalt_dir'])

import libsalt

state_weight = 1
reward_weight = 1
addTime = 1
control_cycle = 3

from config import TRAIN_CONFIG

IS_DOCKERIZE = TRAIN_CONFIG['IS_DOCKERIZE']

if IS_DOCKERIZE:
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



def H(s):
    probabilities = [n_x/len(s) for x,n_x in collections.Counter(s).items()]
    e_x = [-p_x*math.log(p_x,2) for p_x in probabilities]
    return sum(e_x)


class SALT_doan_multi_PSA(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.state_weight = state_weight
        self.reward_weight = reward_weight
        self.addTime = addTime
        self.reward_func = args.reward_func
        self.actionT = args.action_t
        self.logprint = args.logprint
        self.args = args

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
            if 0:
                _, _, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args)
            else:
                edge_file_path = "magic/doan_20211207.edg.xml"
                tss_file_path = "magic/doan_20211207.tss.xml"
            tree = parse(tss_file_path)
        else:
            # self.salt_scenario = self.dest_dir + 'doan_2021_actionT{}.scenario.json'.format(self.actionT)
            self.salt_scenario = self.dest_dir + 'doan_20211207.scenario.json'
            tree = parse(os.getcwd() + '/data/envs/salt/doan/doan_20211207.tss.xml')

        root = tree.getroot()

        trafficSignal = root.findall("trafficSignal")

        self.target_tl_obj = {}
        self.phase_numbers = []
        i=0

        self.targetList_input = args.target_TL.split(',')
        self.targetList_input2 = []

        for tl_i in self.targetList_input:
            self.targetList_input2.append(tl_i)                         ## ex. SA 101
            self.targetList_input2.append(tl_i.split(" ")[1])           ## ex.101
            self.targetList_input2.append(tl_i.replace(" ", ""))        ## ex. SA101

        for x in trafficSignal:
            if x.attrib['signalGroup'] in self.targetList_input2:
                self.target_tl_obj[x.attrib['nodeID']] = {}
                self.target_tl_obj[x.attrib['nodeID']]['crossName'] = x.attrib['crossName']
                self.target_tl_obj[x.attrib['nodeID']]['signalGroup'] = x.attrib['signalGroup']
                self.target_tl_obj[x.attrib['nodeID']]['offset'] = int(x.find('schedule').attrib['offset'])
                self.target_tl_obj[x.attrib['nodeID']]['minDur'] = [int(y.attrib['minDur']) if 'minDur' in y.attrib else int(y.attrib['duration']) for
                                                                     y in x.findall("schedule/phase")]
                self.target_tl_obj[x.attrib['nodeID']]['maxDur'] = [int(y.attrib['maxDur']) if 'maxDur' in y.attrib else int(y.attrib['duration']) for
                                                                     y in x.findall("schedule/phase")]
                self.target_tl_obj[x.attrib['nodeID']]['cycle'] = np.sum([int(y.attrib['duration']) for y in x.findall("schedule/phase")])
                self.target_tl_obj[x.attrib['nodeID']]['duration'] = [int(y.attrib['duration']) for y in x.findall("schedule/phase")]
                tmp_duration_list = np.array([int(y.attrib['duration']) for y in x.findall("schedule/phase")])
                # self.target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(tmp_duration_list > 5)
                self.target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(np.array(self.target_tl_obj[x.attrib['nodeID']]['minDur'])!=np.array(self.target_tl_obj[x.attrib['nodeID']]['maxDur']))
                # print(self.target_tl_obj[x.attrib['nodeID']]['green_idx'])
                self.target_tl_obj[x.attrib['nodeID']]['main_green_idx'] = np.where(tmp_duration_list==np.max(tmp_duration_list))
                self.target_tl_obj[x.attrib['nodeID']]['sub_green_idx'] = list(set(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0]) - set(np.where(tmp_duration_list==np.max(tmp_duration_list))[0]))
                self.target_tl_obj[x.attrib['nodeID']]['tl_idx'] = i
                self.target_tl_obj[x.attrib['nodeID']]['remain'] = self.target_tl_obj[x.attrib['nodeID']]['cycle'] - np.sum(self.target_tl_obj[x.attrib['nodeID']]['minDur'])
                #print(len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0]), self.target_tl_obj[x.attrib['nodeID']]['main_green_idx'][0][0])
                self.target_tl_obj[x.attrib['nodeID']]['max_phase'] = np.where(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0] == self.target_tl_obj[x.attrib['nodeID']]['main_green_idx'][0][0])
                # self.target_tl_obj[x.attrib['nodeID']]['action_list'] = getActionList(len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0]), self.target_tl_obj[x.attrib['nodeID']]['max_phase'][0][0])
                # # self.target_tl_obj[x.attrib['nodeID']]['action_space'] = ((len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0])-1)*2) + 1
                # self.target_tl_obj[x.attrib['nodeID']]['action_space'] = len(self.target_tl_obj[x.attrib['nodeID']]['action_list'])
                self.target_tl_obj[x.attrib['nodeID']]['action_space'] = len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0])
                self.phase_numbers.append(len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0]))
                i+=1

        self.max_phase_length = int(np.max(self.phase_numbers))

        self.target_tl_id_list = list(self.target_tl_obj.keys())
        self.agent_num = len(self.target_tl_id_list)

        self.action_mask = np.zeros(self.agent_num)
        self.control_cycle = control_cycle

        print('target tl obj {}'.format(self.target_tl_obj))
        print('target tl id list {}'.format(self.target_tl_id_list))
        print('number of target tl {}'.format(len(self.target_tl_id_list)))

        if IS_DOCKERIZE:
            tree = parse(edge_file_path)
        else:
            tree = parse(os.getcwd() + '/data/envs/salt/doan/doan_20210401.edg.xml')

        root = tree.getroot()

        edge = root.findall("edge")

        self.near_tl_obj = {}
        for i in self.target_tl_id_list:
            self.near_tl_obj[i] = {}
            self.near_tl_obj[i]['in_edge_list'] = []
            self.near_tl_obj[i]['in_edge_list_0'] = []
            self.near_tl_obj[i]['in_edge_list_1'] = []
            # self.near_tl_obj[i]['near_length_list'] = []

        for x in edge:
            if x.attrib['to'] in self.target_tl_id_list:
                self.near_tl_obj[x.attrib['to']]['in_edge_list'].append(x.attrib['id'])
                self.near_tl_obj[x.attrib['to']]['in_edge_list_0'].append(x.attrib['id'])

        _edge_len = []
        for n in self.near_tl_obj:
            _tmp_in_edge_list = self.near_tl_obj[n]['in_edge_list']
            _tmp_near_juction_list = []
            for x in edge:
                if x.attrib['id'] in _tmp_in_edge_list:
                    _tmp_near_juction_list.append(x.attrib['from'])

            for x in edge:
                if x.attrib['to'] in _tmp_near_juction_list:
                    self.near_tl_obj[n]['in_edge_list'].append(x.attrib['id'])
                    self.near_tl_obj[n]['in_edge_list_1'].append(x.attrib['id'])

            self.target_tl_obj[n]['in_edge_list'] = self.near_tl_obj[n]['in_edge_list']
            self.target_tl_obj[n]['in_edge_list_0'] = self.near_tl_obj[n]['in_edge_list_0']
            self.target_tl_obj[n]['in_edge_list_1'] = self.near_tl_obj[n]['in_edge_list_1']
            _edge_len.append(len(self.near_tl_obj[n]['in_edge_list']))

        self.max_edge_length = int(np.max(_edge_len))
        print(self.target_tl_obj)
        print(self.max_edge_length)

        self.done = False

        libsalt.start(self.salt_scenario)
        libsalt.setCurrentStep(self.startStep)

        print("init", [libsalt.trafficsignal.getTLSConnectedLinkID(x) for x in self.target_tl_id_list])
        print("init", [libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(x) for x in self.target_tl_id_list])
        print("init", [libsalt.trafficsignal.getLastTLSPhaseSwitchingTimeByNodeID(x) for x in self.target_tl_id_list])
        print("init", [len(libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(x).myPhaseVector) for x in self.target_tl_id_list])
        print("init", [libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(x).myPhaseVector[0][1] for x in self.target_tl_id_list])


        _lane_len = []
        for target in self.target_tl_obj:
            _lane_list = []
            _lane_list_0 = []
            for edge in self.target_tl_obj[target]['in_edge_list_0']:
                for lane in range(libsalt.link.getNumLane(edge)):
                    _lane_id = "{}_{}".format(edge, lane)
                    _lane_list.append(_lane_id)
                    _lane_list_0.append((_lane_id))
                    # print(_lane_id, libsalt.lane.getLength(_lane_id))
            self.target_tl_obj[target]['in_lane_list_0'] = _lane_list_0
            _lane_list_1 = []
            for edge in self.target_tl_obj[target]['in_edge_list_1']:
                for lane in range(libsalt.link.getNumLane(edge)):
                    _lane_id = "{}_{}".format(edge, lane)
                    _lane_list.append(_lane_id)
                    _lane_list_1.append((_lane_id))
                    # print(_lane_id, libsalt.lane.getLength(_lane_id))
            self.target_tl_obj[target]['in_lane_list_1'] = _lane_list_1
            self.target_tl_obj[target]['in_lane_list'] = _lane_list
            if self.args.state == 'vd':
                self.target_tl_obj[target]['state_space'] = len(_lane_list)*2 + 1
            else:
                self.target_tl_obj[target]['state_space'] = len(_lane_list) + 1
            _lane_len.append(len(_lane_list))
        self.max_lane_length = np.max(_lane_len)
        print(self.target_tl_obj)
        print(np.max(_lane_len))
        self._observation = []
        # for i in range(len(self.min_green)):
        #     self._observation = np.append(self._observation, self.get_state(self.ts_ids[0]))
        self._observation = np.append(self._observation, self.get_state(self.target_tl_id_list[0]))

        self.obs_len = len(self._observation)

        self.observations = []
        self.lane_passed = []
        for target in self.target_tl_obj:
            self.observations.append([0]*self.target_tl_obj[target]['state_space'])
            self.lane_passed.append([])
        # print("self.lane_passed", self.lane_passed)
        # print(observations)
        self.rewards = np.zeros(self.agent_num)
        self.before_action = np.zeros(self.agent_num)

        libsalt.close()
        # print('{} traci closed\n'.format(self.uid))

        self.simulationSteps = 0

    def step(self, actions):
        self.done = False
        # print('step')

        currentStep = libsalt.getCurrentStep()
        self.simulationSteps = currentStep
        ### change to yellow or keep current green phase
        for i in range(len(self.target_tl_id_list)):
            tlid = self.target_tl_id_list[i]
            scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
            phase_length = len(self.target_tl_obj[tlid]['duration'])

            green_idx = self.target_tl_obj[tlid]['green_idx'][0]

            action_phase = green_idx[int(self.before_action[i])]

            yPhase = (action_phase + 1) % phase_length
            if self.target_tl_obj[tlid]['duration'][yPhase] > 5:
                yPhase = (action_phase + 2) % phase_length
            # print("tl_name {} before action {} current action {}".format(self.target_tl_obj[tlid]['crossName'], self.before_action[i], actions[i]))
            if self.before_action[i] == actions[i]:
                libsalt.trafficsignal.changeTLSPhase(currentStep, tlid, scheduleID, int(action_phase))
            else:
                libsalt.trafficsignal.changeTLSPhase(currentStep, tlid, scheduleID, int(yPhase))

            # currentPhase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)
            # if self.target_tl_obj[tlid]['crossName'] == '진터네거리':
            # print("step {} tl_name {} before_actions {} action_phase {} getCurrPhaseYellow {} rewards {}".format(
            #     self.simulationSteps, self.target_tl_obj[tlid]['crossName'], self.before_action[i], action_phase,
            #     currentPhase, np.round(self.rewards[i], 2)))

        for i in range(3):
            libsalt.simulationStep()

        currentStep = libsalt.getCurrentStep()
        self.simulationSteps = currentStep
        setPhaseStep = currentStep
        action_phase_list = []
        current_phase_list = []
        for i in range(len(self.target_tl_id_list)):
            tlid = self.target_tl_id_list[i]
            scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
            # currentPhase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)

            green_idx = self.target_tl_obj[tlid]['green_idx'][0]

            action_phase = green_idx[actions[i]]
            action_phase_list = np.append(action_phase_list, int(action_phase))

            libsalt.trafficsignal.changeTLSPhase(currentStep, tlid, scheduleID, int(action_phase))
            currPhase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)
            current_phase_list = np.append(current_phase_list, currPhase)

            self.observations[i] = self.get_state(tlid)

            # self.lane_passed[i] = []

        for i in range(self.actionT):
            libsalt.simulationStep()

        self.simulationSteps = libsalt.getCurrentStep()
        for i in range(len(self.target_tl_id_list)):
            link_list_0 = self.target_tl_obj[self.target_tl_id_list[i]]['in_edge_list_0']
            lane_list_0 = self.target_tl_obj[self.target_tl_id_list[i]]['in_lane_list_0']
            link_list_1 = self.target_tl_obj[self.target_tl_id_list[i]]['in_edge_list_1']
            lane_list_1 = self.target_tl_obj[self.target_tl_id_list[i]]['in_lane_list_1']
            if self.reward_func=='pn':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getSumPassed(l))
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getSumPassed(l) * reward_weight)
                self.rewards[i] = np.sum(self.lane_passed[i])
            if self.reward_func=='wt':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingTime(l)/self.actionT)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingTime(l)/self.actionT * reward_weight)
                self.rewards[i] = -np.sum(self.lane_passed[i])
            if self.reward_func=='wt_max':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingTime(l)/self.actionT)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingTime(l)/self.actionT * reward_weight)
                self.rewards[i] = -np.max(self.lane_passed[i])
            if self.reward_func=='wq':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingQLength(l))
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingQLength(l) * reward_weight)
                self.rewards[i] = -np.sum(self.lane_passed[i])
            if self.reward_func=='wt_SBV':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps)/1000)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps)/1000 * reward_weight)
                self.rewards[i] = -np.sum(self.lane_passed[i])
            if self.reward_func=='wt_SBV_max':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps)/1000)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps)/1000 * reward_weight)
                self.rewards[i] = -np.max(self.lane_passed[i])
            if self.reward_func=='wt_ABV':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulationSteps)/1000)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulationSteps)/1000 * reward_weight)
                self.rewards[i] = -np.sum(self.lane_passed[i])

            self.lane_passed[i] = []


        self.simulationSteps = libsalt.getCurrentStep()

        if self.logprint:
            for i in range(len(self.target_tl_id_list)):
                tlid = self.target_tl_id_list[i]
                print("step {} tl_name {} actions {} action_phase {} getCurrPhaseGreen {} reward {}".format(
                    setPhaseStep, self.target_tl_obj[tlid]['crossName'], np.round(actions[i], 3), action_phase_list[i],
                    current_phase_list[i], np.round(self.rewards[i], 2)))

        if self.simulationSteps > self.endStep:
            self.done = True
            print("self.done step {}".format(self.simulationSteps))
            libsalt.close()

        info = {}
        # print(self.before_action, actions)
        self.before_action = actions.copy()

        return self.observations, self.rewards, self.done, info

    def reset(self):

        libsalt.start(self.salt_scenario)
        libsalt.setCurrentStep(self.startStep)

        self.simulationSteps = libsalt.getCurrentStep()

        observations = []
        self.lane_passed = []
        for target in self.target_tl_obj:
            print(target, self.get_state(target))
            observations.append(self.get_state(target))
            self.lane_passed.append([])
        #
        # print("reset", observations)
        # print(observations.shape)

        return observations

    def get_state(self, tsid):
        densityMatrix = []
        passedMatrix = []
        vddMatrix = []
        tlMatrix = []

        link_list = self.target_tl_obj[tsid]['in_lane_list']
        link_list_0 = self.target_tl_obj[tsid]['in_lane_list_0']
        link_list_1 = self.target_tl_obj[tsid]['in_lane_list_1']

        # tl_s = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tsid).myPhaseVector[0][1]
        #print("tl_s {}".format(tl_s))

        for link in link_list_0:
            if self.args.state=='d':
                densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link))
            if self.args.state=='v':
                passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link))
            if self.args.state=='vd':
                densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link))
                passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link))
            # passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link))
            if self.args.state == 'vdd':
                vddMatrix = np.append(vddMatrix, libsalt.lane.getNumVehPassed(link)/(libsalt.lane.getAverageDensity(link)+sys.float_info.epsilon))
        for link in link_list_1:
            if self.args.state=='d':
                densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link) * self.state_weight)
            if self.args.state=='v':
                passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link) * self.state_weight)
            if self.args.state=='vd':
                densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link) * self.state_weight)
                passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link) * self.state_weight)
            if self.args.state == 'vdd':
                vddMatrix = np.append(vddMatrix, libsalt.lane.getNumVehPassed(link)/(libsalt.lane.getAverageDensity(link)+sys.float_info.epsilon) * self.state_weight)

        tlMatrix = np.append(tlMatrix, libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tsid))
        #print(lightMatrix)
        if self.args.state == 'd':
            obs = np.append(densityMatrix, tlMatrix)
        if self.args.state == 'v':
            obs = np.append(passedMatrix, tlMatrix)
        if self.args.state == 'vd':
            obs = np.append(densityMatrix, passedMatrix)
            obs = np.append(obs, tlMatrix)
        if self.args.state == 'vdd':
            obs = np.append(vddMatrix, tlMatrix)

        # print(densityMatrix)
        # print(passedMatrix)

        return obs

    def render(self, mode='human'):
        pass
        # print(self.reward)

    def close(self):
        libsalt.close()
        print('close')

### state -> link to lane 1-hop, passedNum, density(fundamental diagram)
### action -> add or sub to sub phase from max phase with changed phase(every 5 cycle)
### reward -> lane waitingQLength, penalty if phase_sum is not equal to cycle
### reward -> get each 10 step and sum
class SALT_doan_multi_PSA_test(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        self.state_weight = state_weight
        self.reward_weight = reward_weight
        self.addTime = addTime

        self.reward_func = args.reward_func
        self.actionT = args.action_t


        if IS_DOCKERIZE:
            scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)

            # self.startStep = args.testStartTime if args.testStartTime > scenario_begin else scenario_begin
            # self.endStep = args.testEndTime if args.testEndTime < scenario_end else scenario_end
            self.startStep = args.start_time if args.start_time > scenario_begin else scenario_begin
            self.endStep = args.end_time if args.end_time < scenario_end else scenario_end
            self.args = args

        else:
            self.startStep = args.testStartTime
            self.endStep = args.testEndTime
            self.args = args

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.uid = str(uuid.uuid4())

        if IS_DOCKERIZE:
            abs_scenario_file_path = '{}/{}'.format(os.getcwd(), self.args.scenario_file_path)
            self.src_dir = os.path.dirname(abs_scenario_file_path)
            self.dest_dir = os.path.split(self.src_dir)[0]
            self.dest_dir = '{}/data/{}/'.format(self.dest_dir, self.uid)
            os.makedirs(self.dest_dir, exist_ok=True)
        else:
            self.src_dir = os.getcwd() + "/data/envs/salt/doan"
            self.dest_dir = os.getcwd() + "/data/envs/salt/data/" + self.uid + "/"
            os.mkdir(self.dest_dir)

        src_files = os.listdir(self.src_dir)
        for file_name in src_files:
            full_file_name = os.path.join(self.src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, self.dest_dir)

        if IS_DOCKERIZE:
            scenario_file_name = self.args.scenario_file_path.split('/')[-1]
            self.salt_scenario = "{}/{}".format(self.dest_dir, scenario_file_name)

            if 0:
                _, _, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args)
            else:
                edge_file_path = "magic/doan_20210401.edg.xml"
                tss_file_path = "magic/doan(without dan).tss.xml"

            tree = parse(tss_file_path)

        else:
            self.salt_scenario = self.dest_dir + 'doan_2021_test.scenario.json'

            tree = parse(os.getcwd() + '/data/envs/salt/doan/doan(without dan).tss.xml')

        root = tree.getroot()

        trafficSignal = root.findall("trafficSignal")

        self.target_tl_obj = {}
        self.phase_numbers = []
        i=0
        if IS_DOCKERIZE:
            self.targetList_input = args.target_TL.split(',')
        else:
            self.targetList_input = args.targetTL.split(',')

        self.targetList_input2 = []

        for tl_i in self.targetList_input:
            self.targetList_input2.append(tl_i)                         ## ex. SA 101
            self.targetList_input2.append(tl_i.split(" ")[1])           ## ex.101
            self.targetList_input2.append(tl_i.replace(" ", ""))        ## ex. SA101

        for x in trafficSignal:
            if x.attrib['signalGroup'] in self.targetList_input2:
                self.target_tl_obj[x.attrib['nodeID']] = {}
                self.target_tl_obj[x.attrib['nodeID']]['crossName'] = x.attrib['crossName']
                self.target_tl_obj[x.attrib['nodeID']]['signalGroup'] = x.attrib['signalGroup']
                self.target_tl_obj[x.attrib['nodeID']]['offset'] = int(x.find('schedule').attrib['offset'])
                self.target_tl_obj[x.attrib['nodeID']]['minDur'] = [int(y.attrib['minDur']) if 'minDur' in y.attrib else int(y.attrib['duration']) for
                                                                     y in x.findall("schedule/phase")]
                self.target_tl_obj[x.attrib['nodeID']]['maxDur'] = [int(y.attrib['maxDur']) if 'maxDur' in y.attrib else int(y.attrib['duration']) for
                                                                     y in x.findall("schedule/phase")]
                self.target_tl_obj[x.attrib['nodeID']]['cycle'] = np.sum([int(y.attrib['duration']) for y in x.findall("schedule/phase")])
                self.target_tl_obj[x.attrib['nodeID']]['duration'] = [int(y.attrib['duration']) for y in x.findall("schedule/phase")]
                tmp_duration_list = np.array([int(y.attrib['duration']) for y in x.findall("schedule/phase")])
                # self.target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(tmp_duration_list > 5)
                self.target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(np.array(self.target_tl_obj[x.attrib['nodeID']]['minDur'])!=np.array(self.target_tl_obj[x.attrib['nodeID']]['maxDur']))
                # print(self.target_tl_obj[x.attrib['nodeID']]['green_idx'])
                self.target_tl_obj[x.attrib['nodeID']]['main_green_idx'] = np.where(tmp_duration_list==np.max(tmp_duration_list))
                self.target_tl_obj[x.attrib['nodeID']]['sub_green_idx'] = list(set(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0]) - set(np.where(tmp_duration_list==np.max(tmp_duration_list))[0]))
                self.target_tl_obj[x.attrib['nodeID']]['tl_idx'] = i
                self.target_tl_obj[x.attrib['nodeID']]['remain'] = self.target_tl_obj[x.attrib['nodeID']]['cycle'] - np.sum(self.target_tl_obj[x.attrib['nodeID']]['minDur'])
                #print(len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0]), self.target_tl_obj[x.attrib['nodeID']]['main_green_idx'][0][0])
                self.target_tl_obj[x.attrib['nodeID']]['max_phase'] = np.where(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0] == self.target_tl_obj[x.attrib['nodeID']]['main_green_idx'][0][0])
                # self.target_tl_obj[x.attrib['nodeID']]['action_list'] = getActionList(len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0]), self.target_tl_obj[x.attrib['nodeID']]['max_phase'][0][0])
                # # self.target_tl_obj[x.attrib['nodeID']]['action_space'] = ((len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0])-1)*2) + 1
                # self.target_tl_obj[x.attrib['nodeID']]['action_space'] = len(self.target_tl_obj[x.attrib['nodeID']]['action_list'])
                self.target_tl_obj[x.attrib['nodeID']]['action_space'] = len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0])
                self.phase_numbers.append(len(self.target_tl_obj[x.attrib['nodeID']]['green_idx'][0]))
                i+=1

        self.max_phase_length = int(np.max(self.phase_numbers))

        self.target_tl_id_list = list(self.target_tl_obj.keys())
        self.agent_num = len(self.target_tl_id_list)

        self.action_mask = np.zeros(self.agent_num)
        self.control_cycle = control_cycle


        print('target tl obj {}'.format(self.target_tl_obj))
        print('target tl id list {}'.format(self.target_tl_id_list))
        print('number of target tl {}'.format(len(self.target_tl_id_list)))

        if IS_DOCKERIZE:
            tree = parse(edge_file_path)
        else:
            tree = parse(os.getcwd() + '/data/envs/salt/doan/doan_20210401.edg.xml')

        root = tree.getroot()

        edge = root.findall("edge")

        self.near_tl_obj = {}

        if IS_DOCKERIZE:
            if self.args.mode=='test':
                self.fn_rl_phase_reward_output = "{}/output/test/rl_phase_reward_output.txt".format(args.io_home)
                f = open(self.fn_rl_phase_reward_output, mode='w+', buffering=-1, encoding='utf-8', errors=None,
                         newline=None, closefd=True, opener=None)
                f.write('step,tl_name,actions,phase,reward\n')
                f.close()
        else:
            if self.args.mode=='test':
                f = open("output/test/rl_phase_reward_output.txt", mode='w+', buffering=-1, encoding='utf-8', errors=None,
                         newline=None,
                         closefd=True, opener=None)

                f.write('step,tl_name,actions,phase,reward\n')
                f.close()
        for i in self.target_tl_id_list:
            self.near_tl_obj[i] = {}
            self.near_tl_obj[i]['in_edge_list'] = []
            self.near_tl_obj[i]['in_edge_list_0'] = []
            self.near_tl_obj[i]['in_edge_list_1'] = []
            # self.near_tl_obj[i]['near_length_list'] = []

        for x in edge:
            if x.attrib['to'] in self.target_tl_id_list:
                self.near_tl_obj[x.attrib['to']]['in_edge_list'].append(x.attrib['id'])
                self.near_tl_obj[x.attrib['to']]['in_edge_list_0'].append(x.attrib['id'])

        _edge_len = []
        for n in self.near_tl_obj:
            _tmp_in_edge_list = self.near_tl_obj[n]['in_edge_list']
            _tmp_near_juction_list = []
            for x in edge:
                if x.attrib['id'] in _tmp_in_edge_list:
                    _tmp_near_juction_list.append(x.attrib['from'])

            for x in edge:
                if x.attrib['to'] in _tmp_near_juction_list:
                    self.near_tl_obj[n]['in_edge_list'].append(x.attrib['id'])
                    self.near_tl_obj[n]['in_edge_list_1'].append(x.attrib['id'])

            self.target_tl_obj[n]['in_edge_list'] = self.near_tl_obj[n]['in_edge_list']
            self.target_tl_obj[n]['in_edge_list_0'] = self.near_tl_obj[n]['in_edge_list_0']
            self.target_tl_obj[n]['in_edge_list_1'] = self.near_tl_obj[n]['in_edge_list_1']
            _edge_len.append(len(self.near_tl_obj[n]['in_edge_list']))

        self.max_edge_length = int(np.max(_edge_len))
        print(self.target_tl_obj)
        print(self.max_edge_length)

        self.done = False

        libsalt.start(self.salt_scenario)
        libsalt.setCurrentStep(self.startStep)

        print("init", [libsalt.trafficsignal.getTLSConnectedLinkID(x) for x in self.target_tl_id_list])
        print("init", [libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(x) for x in self.target_tl_id_list])
        print("init", [libsalt.trafficsignal.getLastTLSPhaseSwitchingTimeByNodeID(x) for x in self.target_tl_id_list])
        print("init", [len(libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(x).myPhaseVector) for x in self.target_tl_id_list])
        print("init", [libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(x).myPhaseVector[0][1] for x in self.target_tl_id_list])


        _lane_len = []
        for target in self.target_tl_obj:
            _lane_list = []
            _lane_list_0 = []
            for edge in self.target_tl_obj[target]['in_edge_list_0']:
                for lane in range(libsalt.link.getNumLane(edge)):
                    _lane_id = "{}_{}".format(edge, lane)
                    _lane_list.append(_lane_id)
                    _lane_list_0.append((_lane_id))
                    # print(_lane_id, libsalt.lane.getLength(_lane_id))
            self.target_tl_obj[target]['in_lane_list_0'] = _lane_list_0
            _lane_list_1 = []
            for edge in self.target_tl_obj[target]['in_edge_list_1']:
                for lane in range(libsalt.link.getNumLane(edge)):
                    _lane_id = "{}_{}".format(edge, lane)
                    _lane_list.append(_lane_id)
                    _lane_list_1.append((_lane_id))
                    # print(_lane_id, libsalt.lane.getLength(_lane_id))
            self.target_tl_obj[target]['in_lane_list_1'] = _lane_list_1
            self.target_tl_obj[target]['in_lane_list'] = _lane_list
            if self.args.state == 'vd':
                self.target_tl_obj[target]['state_space'] = len(_lane_list)*2 + 1
            else:
                self.target_tl_obj[target]['state_space'] = len(_lane_list) + 1
            _lane_len.append(len(_lane_list))
        self.max_lane_length = np.max(_lane_len)
        print(self.target_tl_obj)
        print(np.max(_lane_len))
        self._observation = []
        # for i in range(len(self.min_green)):
        #     self._observation = np.append(self._observation, self.get_state(self.ts_ids[0]))
        self._observation = np.append(self._observation, self.get_state(self.target_tl_id_list[0]))

        self.obs_len = len(self._observation)

        self.observations = []
        self.lane_passed = []
        for target in self.target_tl_obj:
            self.observations.append([0]*self.target_tl_obj[target]['state_space'])
            self.lane_passed.append([])
        print("self.lane_passed", self.lane_passed)
        # print(observations)
        self.rewards = np.zeros(self.agent_num)
        self.before_action = np.zeros(self.agent_num)

        libsalt.close()
        print('{} traci closed\n'.format(self.uid))

        self.simulationSteps = 0


    def step(self, actions):
        self.done = False
        # print('step')

        currentStep = libsalt.getCurrentStep()
        self.simulationSteps = currentStep
        ### change to yellow or keep current green phase
        for i in range(len(self.target_tl_id_list)):
            tlid = self.target_tl_id_list[i]
            scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
            phase_length = len(self.target_tl_obj[tlid]['duration'])

            green_idx = self.target_tl_obj[tlid]['green_idx'][0]

            action_phase = green_idx[int(self.before_action[i])]

            yPhase = (action_phase + 1) % phase_length
            if self.target_tl_obj[tlid]['duration'][yPhase] > 5:
                yPhase = (action_phase + 2) % phase_length
            # print("tl_name {} before action {} current action {}".format(self.target_tl_obj[tlid]['crossName'], self.before_action[i], actions[i]))
            if self.before_action[i] == actions[i]:
                libsalt.trafficsignal.changeTLSPhase(currentStep, tlid, scheduleID, int(action_phase))
            else:
                libsalt.trafficsignal.changeTLSPhase(currentStep, tlid, scheduleID, int(yPhase))

            # currentPhase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)
            # if self.target_tl_obj[tlid]['crossName'] == '진터네거리':
            # print("step {} tl_name {} before_actions {} action_phase {} getCurrPhaseYellow {} rewards {}".format(
            #     self.simulationSteps, self.target_tl_obj[tlid]['crossName'], self.before_action[i], action_phase,
            #     currentPhase, np.round(self.rewards[i], 2)))

        for i in range(3):
            libsalt.simulationStep()

        currentStep = libsalt.getCurrentStep()
        self.simulationSteps = currentStep
        setPhaseStep = currentStep
        action_phase_list = []
        current_phase_list = []
        for i in range(len(self.target_tl_id_list)):
            tlid = self.target_tl_id_list[i]
            scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
            # currentPhase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)

            green_idx = self.target_tl_obj[tlid]['green_idx'][0]

            action_phase = green_idx[actions[i]]
            action_phase_list = np.append(action_phase_list, int(action_phase))

            libsalt.trafficsignal.changeTLSPhase(currentStep, tlid, scheduleID, int(action_phase))
            currPhase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)
            current_phase_list = np.append(current_phase_list, currPhase)

            self.observations[i] = self.get_state(tlid)

            # self.lane_passed[i] = []

        for i in range(self.actionT):
            libsalt.simulationStep()

        self.simulationSteps = libsalt.getCurrentStep()
        for i in range(len(self.target_tl_id_list)):
            link_list_0 = self.target_tl_obj[self.target_tl_id_list[i]]['in_edge_list_0']
            link_list_1 = self.target_tl_obj[self.target_tl_id_list[i]]['in_edge_list_1']
            if self.reward_func=='pn':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getSumPassed(l))
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getSumPassed(l) * reward_weight)
                self.rewards[i] = np.sum(self.lane_passed[i])
            if self.reward_func=='wt':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingTime(l)/self.actionT)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingTime(l)/self.actionT * reward_weight)
                self.rewards[i] = -np.sum(self.lane_passed[i])
            if self.reward_func=='wt_max':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingTime(l)/self.actionT)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingTime(l)/self.actionT * reward_weight)
                self.rewards[i] = -np.max(self.lane_passed[i])
            if self.reward_func=='wq':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingQLength(l))
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getAverageWaitingQLength(l) * reward_weight)
                self.rewards[i] = -np.sum(self.lane_passed[i])
            if self.reward_func=='wt_SBV':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps)/1000)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps)/1000 * reward_weight)
                self.rewards[i] = -np.sum(self.lane_passed[i])
            if self.reward_func=='wt_SBV_max':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps)/1000)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentWaitingTimeSumBaseVehicle(l, self.simulationSteps)/1000 * reward_weight)
                self.rewards[i] = -np.max(self.lane_passed[i])
            if self.reward_func=='wt_ABV':
                for l in link_list_0:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulationSteps)/1000)
                for l in link_list_1:
                    self.lane_passed[i] = np.append(self.lane_passed[i], libsalt.link.getCurrentAverageWaitingTimeBaseVehicle(l, self.simulationSteps)/1000 * reward_weight)
                self.rewards[i] = -np.sum(self.lane_passed[i])


            self.lane_passed[i] = []


        self.simulationSteps = libsalt.getCurrentStep()


        if IS_DOCKERIZE:
            if self.args.mode=='test':
                f = open(self.fn_rl_phase_reward_output, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                         newline=None,
                         closefd=True, opener=None)
                for i in range(len(self.target_tl_id_list)):
                    tlid = self.target_tl_id_list[i]
                    f.write("{},{},{},{},{}\n".format(setPhaseStep, self.target_tl_obj[tlid]['crossName'], actions[i],
                                                      action_phase_list[i], np.round(self.rewards[i], 2)))
                    print("step {} tl_name {} action {} phase {} reward {}\n".format(setPhaseStep, self.target_tl_obj[tlid]['crossName'], actions[i],
                                                      action_phase_list[i], np.round(self.rewards[i], 2)))
                f.close()

        else:
            if self.args.mode=='test':
                f = open("output/test/rl_phase_reward_output.txt", mode='a+', buffering=-1, encoding='utf-8', errors=None,
                         newline=None,
                         closefd=True, opener=None)

                for i in range(len(self.target_tl_id_list)):
                    tlid = self.target_tl_id_list[i]
                    f.write("{},{},{},{},{}\n".format(setPhaseStep, self.target_tl_obj[tlid]['crossName'], actions[i],
                                                      action_phase_list[i], np.round(self.rewards[i], 2)))
                    print("step {} tl_name {} action {} phase {} reward {}\n".format(setPhaseStep, self.target_tl_obj[tlid]['crossName'], actions[i],
                                                      action_phase_list[i], np.round(self.rewards[i], 2)))
                f.close()

        if self.simulationSteps > self.endStep:
            self.done = True
            print("self.done step {}".format(self.simulationSteps))
            libsalt.close()

        info = {}
        # print(self.before_action, actions)
        self.before_action = actions.copy()

        return self.observations, self.rewards, self.done, info

    def reset(self):
        self.uid = str(uuid.uuid4())
        if IS_DOCKERIZE:
            self.dest_dir = os.path.split(self.src_dir)[0]
            self.dest_dir = '{}/data/{}/'.format(self.dest_dir, self.uid)
            os.makedirs(self.dest_dir, exist_ok=True)
        else:
            self.dest_dir = os.getcwd() + "/data/envs/salt/data/" + self.uid + "/"
            os.mkdir(self.dest_dir)

        src_files = os.listdir(self.src_dir)
        for file_name in src_files:
            full_file_name = os.path.join(self.src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, self.dest_dir)

        if IS_DOCKERIZE:
            scenario_file_name = self.args.scenario_file_path.split('/')[-1]
            self.salt_scenario = "{}/{}".format(self.dest_dir, scenario_file_name)
        else:
            self.salt_scenario = self.dest_dir + 'doan_2021_test.scenario.json'

        libsalt.start(self.salt_scenario)
        libsalt.setCurrentStep(self.startStep)

        self.simulationSteps = libsalt.getCurrentStep()

        observations = []
        self.lane_passed = []
        for target in self.target_tl_obj:
            print(target, self.get_state(target))
            observations.append(self.get_state(target))
            self.lane_passed.append([])
        #
        # print("reset", observations)
        # print(observations.shape)

        return observations

    def get_state(self, tsid):
        densityMatrix = []
        passedMatrix = []
        tlMatrix = []

        link_list = self.target_tl_obj[tsid]['in_lane_list']
        link_list_0 = self.target_tl_obj[tsid]['in_lane_list_0']
        link_list_1 = self.target_tl_obj[tsid]['in_lane_list_1']

        # tl_s = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tsid).myPhaseVector[0][1]
        #print("tl_s {}".format(tl_s))

        for link in link_list_0:
            if self.args.state=='d':
                densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link))
            if self.args.state=='v':
                passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link))
            if self.args.state=='vd':
                densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link))
                passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link))
            # passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link))
        for link in link_list_1:
            if self.args.state=='d':
                densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link) * self.state_weight)
            if self.args.state=='v':
                passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link) * self.state_weight)
            if self.args.state=='vd':
                densityMatrix = np.append(densityMatrix, libsalt.lane.getAverageDensity(link) * self.state_weight)
                passedMatrix = np.append(passedMatrix, libsalt.lane.getNumVehPassed(link) * self.state_weight)

        tlMatrix = np.append(tlMatrix, libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tsid))
        #print(lightMatrix)
        if self.args.state == 'd':
            obs = np.append(densityMatrix, tlMatrix)
        if self.args.state == 'v':
            obs = np.append(passedMatrix, tlMatrix)
        if self.args.state == 'vd':
            obs = np.append(densityMatrix, passedMatrix)
            obs = np.append(obs, tlMatrix)

        # print(densityMatrix)
        # print(passedMatrix)

        return obs

    def render(self, mode='human'):
        pass
        # print(self.reward)

    def close(self):
        libsalt.close()
        print('close')

