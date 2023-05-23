# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import platform
import pprint
import shutil
import uuid
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
from deprecated import deprecated

import libsalt

from DebugConfiguration import DBG_OPTIONS
from TSOConstants import _REWARD_GATHER_UNIT_, _RESULT_COMP_



def getScenarioRelatedFilePath(scenario_file_path):
    '''
    get node-, edge-, tss-file path from scenario file
    :param scenario_file_path:
    :return:
    '''
    abs_scenario_file_path = '{}/{}'.format(os.getcwd(), scenario_file_path)

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
    '''
    get begin- & end-time of scenario from scenario file
    :param scenario_file_path:
    :return:
    '''
    abs_scenario_file_path = '{}/{}'.format(os.getcwd(), scenario_file_path)

    with open(abs_scenario_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        begin_time = json_data["scenario"]["time"]["begin"]
        end_time = json_data["scenario"]["time"]["end"]

    return begin_time, end_time



def getSimulationStartStepAndEndStep(args):
    '''
    get begin- & end-time of simulation

    :param args:
    :return:
    '''
    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_step = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_step = args.end_time if args.end_time < scenario_end else scenario_end
    return start_step, end_step



def makePosssibleSaNameList(sa_names):
    '''
    get possible SA names which indicate same SA
    :param sa_names:
    :return:
    '''
    cvted_sa_name_list = []
    in_sa = sa_names.split(',')
    for t in in_sa:
        t = t.strip()   # remove  any leading and trailing white space
        cvted_sa_name_list.append(t)   ## ex. SA 101
        cvted_sa_name_list.append(t.split(" ")[1])  ## ex.101
        cvted_sa_name_list.append(t.replace(" ", ""))  ## ex. SA101

    return cvted_sa_name_list



def copyScenarioFiles(scenario_file_path):
    '''
    copy scenario related files and return copied path
    :param scenario_file_path:
    :return:
    '''
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    uid = str(uuid.uuid4())

    abs_scenario_file_path = '{}/{}'.format(os.getcwd(), scenario_file_path)
    src_dir = os.path.dirname(abs_scenario_file_path)
    dest_dir = os.path.split(src_dir)[0]
    dest_dir = '{}/data/{}/'.format(dest_dir, uid)
    os.makedirs(dest_dir, exist_ok=True)

    src_files = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)

    scenario_file_name = scenario_file_path.split('/')[-1]
    salt_scenario = "{}/{}".format(dest_dir, scenario_file_name)

    return salt_scenario


### 녹색 시간 조정 actioon 생성
def getActionList(phase_num, max_phase):
    '''
    create list of possible actions which can ge used to adjust green time
    :param phase_num:
    :param max_phase:
    :return:
    '''
    _pos = [4, 3, 2, 1, 0, -1, -2, -3, -4]

    phase_num = phase_num
    max_phase = max_phase
    mask = np.ones(phase_num, dtype=bool)
    mask[max_phase] = 0
    if phase_num <= 5:
        if phase_num == 2:
            meshgrid = np.array(np.meshgrid(_pos, _pos)).T.reshape(-1, phase_num)
        if phase_num == 3:
            meshgrid = np.array(np.meshgrid(_pos, _pos, _pos)).T.reshape(-1, phase_num)
        if phase_num == 4:
            meshgrid = np.array(np.meshgrid(_pos, _pos, _pos, _pos)).T.reshape(-1, phase_num)
        if phase_num == 5:
            meshgrid = np.array(np.meshgrid(_pos, _pos, _pos, _pos, _pos)).T.reshape(-1, phase_num)

        if phase_num == 1:
            action_list = [[0]]
        else:
            action_list = [x.tolist() for x in meshgrid
                                        if x[max_phase] != 0 and x[max_phase] + np.sum(x[mask]) == 0
                                            and np.min(np.abs(x[mask])) == 0 and np.max(np.abs(x[mask])) == 1
                                            and x[max_phase] != np.min(x[mask]) and x[max_phase] != np.max(x[mask])]
    else:
        meshgrid = np.array(np.meshgrid(_pos, _pos, _pos, _pos, _pos, _pos)).T.reshape(-1, phase_num)
        action_list = [x.tolist() for x in meshgrid
                                        if x[max_phase] != 0 and x[max_phase] + np.sum(x[mask]) == 0
                                            and np.min(np.abs(x[mask])) == 0 and np.max(np.abs(x[mask])) == 1
                                            and x[max_phase] != np.min(x[mask]) - 1 and x[max_phase] != np.max(x[mask]) + 1
                                            and x[max_phase] != np.min(x[mask]) and x[max_phase] != np.max(x[mask])]

    if phase_num != 1:
        tmp = list([1] * phase_num)
        tmp[max_phase] = -(phase_num - 1)
        action_list.append(tmp)

        tmp = list([-1] * phase_num)
        tmp[max_phase] = phase_num - 1
        action_list.append(tmp)

        tmp = list([0] * phase_num)
        action_list.append(tmp)

        action_list.reverse()

        for i in range(1, len(action_list)):
            action_list.append(list(np.array(action_list[i])*2))

    return action_list


@deprecated(reason="use another function : getActionList")
def getActionListV2(phase_num, max_phase):
    '''
    create list of possible actions which can ge used to adjust green time

    :param phase_num:
    :param max_phase:
    :return:
    '''
    _pos = [4, 3, 2, 1, 0, -1, -2, -3, -4]

    phase_num = phase_num
    max_phase = max_phase
    mask = np.ones(phase_num, dtype=bool)
    mask[max_phase] = 0
    if phase_num <= 5:
        if phase_num == 2:
            meshgrid = np.array(np.meshgrid(_pos, _pos)).T.reshape(-1, phase_num)
        if phase_num == 3:
            meshgrid = np.array(np.meshgrid(_pos, _pos, _pos)).T.reshape(-1, phase_num)
        if phase_num == 4:
            meshgrid = np.array(np.meshgrid(_pos, _pos, _pos, _pos)).T.reshape(-1, phase_num)
        if phase_num == 5:
            meshgrid = np.array(np.meshgrid(_pos, _pos, _pos, _pos, _pos)).T.reshape(-1, phase_num)

        action_list = [x.tolist() for x in meshgrid
                                        if x[max_phase] != 0 and x[max_phase] + np.sum(x[mask]) == 0
                                            and np.min(np.abs(x[mask])) == 0 and np.max(np.abs(x[mask])) == 1
                                            and x[max_phase] != np.min(x[mask]) and x[max_phase] != np.max(x[mask])
                                            and x[max_phase]>0]
    else:
        meshgrid = np.array(np.meshgrid(_pos, _pos, _pos, _pos, _pos, _pos)).T.reshape(-1, phase_num)
        action_list = [x.tolist() for x in meshgrid
                                        if x[max_phase] != 0 and x[max_phase] + np.sum(x[mask]) == 0
                                            and np.min(np.abs(x[mask])) == 0 and np.max(np.abs(x[mask])) == 1
                                            and x[max_phase] != np.min(x[mask]) - 1 and x[max_phase] != np.max(x[mask]) + 1
                                            and x[max_phase] != np.min(x[mask]) and x[max_phase] != np.max(x[mask])
                                            and x[max_phase]>0]

    tmp = list([-1] * phase_num)
    tmp[max_phase] = phase_num - 1
    action_list.append(tmp)

    tmp = list([0] * phase_num)
    action_list.append(tmp)

    action_list.reverse()

    for i in range(1, len(action_list)):
        action_list.append(list(np.array(action_list[i])*2))

    return action_list

### 녹색 시간 조정 action list에서 제약 조건 벗어나는 action 제거
def getPossibleActionList(args, duration, min_dur, max_dur, green_idx, actionList):
    '''
    remove actions which violate constraints from action list for adjusting the green time
    :param args: 
    :param duration: 
    :param min_dur: 
    :param max_dur: 
    :param green_idx: 
    :param actionList: 
    :return: 
    '''
    duration = np.array(duration)
    minGreen = np.array(min_dur)
    maxGreen = np.array(max_dur)
    green_idx = np.array(green_idx)

    new_actionList = []

    for action in actionList:
        npsum = 0
        newDur = duration[green_idx] + np.array(action) * args.add_time
        npsum += np.sum(minGreen[green_idx] > newDur)
        npsum += np.sum(maxGreen[green_idx] < newDur)
        if npsum == 0:
            new_actionList.append(action)
    if DBG_OPTIONS.PrintSaRelatedInfo:
        print('len(actionList)', len(actionList), 'len(new_actionList)', len(new_actionList))

    return new_actionList


def getScheduleID(traffic_signal, given_start_time):
    '''
    get schedule id
    :param traffic_signal: traffic signal
    :param given_start_time: given simulation start time
    :return: schedule id
    '''
    all_plan = traffic_signal.findall("TODPlan/plan")

    current_start_time = 0
    idx = -1
    for i in range(len(all_plan)):
        y = all_plan[i]
        next_start_time = int(y.attrib["startTime"])
        if (given_start_time >= current_start_time) and (given_start_time < next_start_time):
            idx = i - 1
            break

    if idx == -1:
        schedule = traffic_signal.find("TODPlan").attrib['defaultPlan']
    else:
        schedule = all_plan[idx].attrib['schedule']

    return schedule



def constructTSSRelatedInfo(args, tss_file_path, sa_name_list):
    '''
    construce TSS related info from given traffic environment
    :paran args : parsed argument
    :param tss_file_path: file path of TSS
    :param sa_name_list: target signal group info
    :return:  an object which contains TSS related info
    '''
    tree = parse(tss_file_path)
    root = tree.getroot()
    traffic_signal = root.findall("trafficSignal")

    target_tl_obj = {}
    i = 0
    for x in traffic_signal:
        sg = x.attrib['signalGroup'].strip()
        if sg in sa_name_list:
            target_tl_obj[x.attrib['nodeID']] = {}
            target_tl_obj[x.attrib['nodeID']]['crossName'] = x.attrib['crossName']

            _signalGroup = x.attrib['signalGroup']
            if "SA" not in _signalGroup:
                _signalGroup = "SA " + _signalGroup
            if "SA " not in _signalGroup:
                _signalGroup = _signalGroup.replace("SA", "SA ")

            target_tl_obj[x.attrib['nodeID']]['signalGroup'] = _signalGroup

            s_id = getScheduleID(x, args.start_time)

            # print(_signalGroup)
            target_tl_obj[x.attrib['nodeID']]['offset'] = int(x.find(f"schedule[@id='{s_id}']").attrib['offset'])
            target_tl_obj[x.attrib['nodeID']]['minDur'] = [int(y.attrib['minDur']) if 'minDur' in y.attrib else int(y.attrib['duration']) for
                                                           y in x.findall(f"schedule[@id='{s_id}']/phase")]
            target_tl_obj[x.attrib['nodeID']]['maxDur'] = [int(y.attrib['maxDur']) if 'maxDur' in y.attrib else int(y.attrib['duration']) for
                                                           y in x.findall(f"schedule[@id='{s_id}']/phase")]
            target_tl_obj[x.attrib['nodeID']]['cycle'] = np.sum([int(y.attrib['duration']) for y in x.findall(f"schedule[@id='{s_id}']/phase")])
            target_tl_obj[x.attrib['nodeID']]['duration'] = [int(y.attrib['duration']) for y in x.findall(f"schedule[@id='{s_id}']/phase")]
            tmp_duration_list = np.array([int(y.attrib['duration']) for y in x.findall(f"schedule[@id='{s_id}']/phase")])
            # target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(tmp_duration_list > 5)
            target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(np.array(target_tl_obj[x.attrib['nodeID']]['minDur']) != np.array(target_tl_obj[x.attrib['nodeID']]['maxDur']))

            ### for select discrete action with the current phase ratio from tanH Prob.
            dur_arr = []
            for g in target_tl_obj[x.attrib['nodeID']]['green_idx'][0]:
                dur_arr.append(target_tl_obj[x.attrib['nodeID']]['duration'][g])
            dur_ratio = dur_arr / np.sum(dur_arr)
            tmp = -1
            dur_bins = []
            for dr in dur_ratio:
                dur_bins.append(tmp + dr * 2)
                tmp += dr * 2
            # print(target_tl_obj[x.attrib['nodeID']]['green_idx'])
            target_tl_obj[x.attrib['nodeID']]['duration_bins'] = dur_bins
            target_tl_obj[x.attrib['nodeID']]['main_green_idx'] = np.where(tmp_duration_list == np.max(tmp_duration_list))
            target_tl_obj[x.attrib['nodeID']]['sub_green_idx'] = list(set(target_tl_obj[x.attrib['nodeID']]['green_idx'][0]) - set(np.where(tmp_duration_list == np.max(tmp_duration_list))[0]))
            target_tl_obj[x.attrib['nodeID']]['tl_idx'] = i
            target_tl_obj[x.attrib['nodeID']]['remain'] = target_tl_obj[x.attrib['nodeID']]['cycle'] - np.sum(target_tl_obj[x.attrib['nodeID']]['minDur'])
            target_tl_obj[x.attrib['nodeID']]['max_phase'] = np.where(target_tl_obj[x.attrib['nodeID']]['green_idx'][0] == target_tl_obj[x.attrib['nodeID']]['main_green_idx'][0][0])
            target_tl_obj[x.attrib['nodeID']]['action_space'] = len(target_tl_obj[x.attrib['nodeID']]['green_idx'][0])
            target_tl_obj[x.attrib['nodeID']]['action_list'] = getPossibleActionList(args, target_tl_obj[x.attrib['nodeID']]['duration'],
                                                                                     target_tl_obj[x.attrib['nodeID']]['minDur'],
                                                                                     target_tl_obj[x.attrib['nodeID']]['maxDur'],
                                                                                     target_tl_obj[x.attrib['nodeID']]['green_idx'],
                                                                                     getActionList(len(target_tl_obj[x.attrib['nodeID']]['green_idx'][0]), target_tl_obj[x.attrib['nodeID']]['max_phase'][0][0]))
            i += 1

    return target_tl_obj



def constructEdgeRelatedInfo(edge_file_path, target_tl_id_list, target_tl_obj):
    '''
    construct EDGE related info from given traffic environment

    :param edge_file_path:  file path of EDGE info
    :param target_tl_id_list: id of target traffic light
    :param target_tl_obj: an object to store constructed EDGE related info
    :return:
    '''
    tree = parse(edge_file_path)
    root = tree.getroot()

    edge = root.findall("edge")

    near_tl_obj = {}
    for i in target_tl_id_list:
        near_tl_obj[i] = {}
        near_tl_obj[i]['in_edge_list'] = []
        near_tl_obj[i]['in_edge_list_0'] = []
        near_tl_obj[i]['in_edge_list_1'] = []
        # near_tl_obj[i]['near_length_list'] = []

    for x in edge:
        if x.attrib['to'] in target_tl_id_list:
            near_tl_obj[x.attrib['to']]['in_edge_list'].append(x.attrib['id'])
            near_tl_obj[x.attrib['to']]['in_edge_list_0'].append(x.attrib['id'])

    _edge_len = []
    for n in near_tl_obj:
        _tmp_in_edge_list = near_tl_obj[n]['in_edge_list']
        _tmp_near_juction_list = []
        for x in edge:
            if x.attrib['id'] in _tmp_in_edge_list:
                _tmp_near_juction_list.append(x.attrib['from'])

        for x in edge:
            if x.attrib['to'] in _tmp_near_juction_list:
                near_tl_obj[n]['in_edge_list'].append(x.attrib['id'])
                near_tl_obj[n]['in_edge_list_1'].append(x.attrib['id'])

        target_tl_obj[n]['in_edge_list'] = near_tl_obj[n]['in_edge_list']
        target_tl_obj[n]['in_edge_list_0'] = near_tl_obj[n]['in_edge_list_0']
        target_tl_obj[n]['in_edge_list_1'] = near_tl_obj[n]['in_edge_list_1']
        _edge_len.append(len(near_tl_obj[n]['in_edge_list']))

    return target_tl_obj



def constructLaneRelatedInfo(args, salt_scenario, target_tl_obj):
    '''
    construct LANE related info from given traffic environment
    :param atgs : parsed argument
    :param salt_scenario: scenario file path
    :param target_tl_obj: an object to store constructed LANE related info
    :return:
    '''
    startStep = 0

    libsalt.start(salt_scenario)
    libsalt.setCurrentStep(startStep)

    _lane_len = []
    for target in target_tl_obj:
        _lane_list = []
        _lane_list_0 = []
        for edge in target_tl_obj[target]['in_edge_list_0']:
            for lane in range(libsalt.link.getNumLane(edge)):
                _lane_id = "{}_{}".format(edge, lane)
                _lane_list.append(_lane_id)
                _lane_list_0.append((_lane_id))
                # print(_lane_id, libsalt.lane.getLength(_lane_id))
        target_tl_obj[target]['in_lane_list_0'] = _lane_list_0
        _lane_list_1 = []
        for edge in target_tl_obj[target]['in_edge_list_1']:
            for lane in range(libsalt.link.getNumLane(edge)):
                _lane_id = "{}_{}".format(edge, lane)
                _lane_list.append(_lane_id)
                _lane_list_1.append((_lane_id))
                # print(_lane_id, libsalt.lane.getLength(_lane_id))
        target_tl_obj[target]['in_lane_list_1'] = _lane_list_1
        target_tl_obj[target]['in_lane_list'] = _lane_list
        if args.state == 'vd':
            target_tl_obj[target]['state_space'] = len(_lane_list_0) * 2 + 1
        else:
            target_tl_obj[target]['state_space'] = len(_lane_list_0) + 1
        _lane_len.append(len(_lane_list))

    libsalt.close()

    return target_tl_obj, _lane_len



### 신호 최적화 대상 교차로 및 교차로 그룹에 대한 정보를 object로 생성
def getSaRelatedInfo(args, sa_name_list, salt_scenario):
    '''
    gather SA related info such as contained TLs, TSS, lane, link,....
    :param args: parsed argument
    :param sa_name_list: list of name of SA which are interesting
    :param salt_scenario: scenario file path
    :return:
    '''
    _, _, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args.scenario_file_path)

    target_tl_obj = constructTSSRelatedInfo(args, tss_file_path, sa_name_list)

    ## get the identifier of target intersection to optimize signal
    target_tl_id_list = list(target_tl_obj.keys())

    ## get EDGE info which are belong to the target intersection group for optimizing signal
    target_tl_obj = constructEdgeRelatedInfo(edge_file_path, target_tl_id_list, target_tl_obj)

    ## build incomming LANE related info by executing the simulator
    target_tl_obj, _lane_len = constructLaneRelatedInfo(args, salt_scenario, target_tl_obj)

    if DBG_OPTIONS.PrintSaRelatedInfo:
        print("target_tl_obj")
        pprint.pprint(target_tl_obj, width=200, compact=True)

    ### for SAPPO Agent ###
    sa_obj = {}
    for tl_obj in target_tl_obj:
        if target_tl_obj[tl_obj]['signalGroup'] not in sa_obj:
            sa_obj[target_tl_obj[tl_obj]['signalGroup']] = {}
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['crossName_list'] = [] # 교차로 그룹에 속한 교차로 이름 목록
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tlid_list'] = []      # 교차로 id 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space'] = 0     # state space - 교차로 그룹에 속한 교차로의 in_lane 수
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space'] = 0    # action space - 교차로 그룹에 속한 교차로 수 : gro의 경우는 2배
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'] = []     # action_min에 대한 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'] = []     # action_max에 대한 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['offset_list'] = []    # 각 교차로의 offset 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['minDur_list'] = []    # 각 교차로의 최소 녹색 시간 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['maxDur_list'] = []    # 각 교차로의 최대 녹색 시간 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['cycle_list'] = []     # 각 교차로의 주기 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_list'] = []  # 각 교차로의 현재 신호 시간 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['green_idx_list'] = [] # 각 교차로의 녹색 시간 인덱스 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_bins_list'] = []     # 각 교차로의 현재 신호 비율에 따라 -1에서 1까지 등분한 리스트(ppo phase 선택 action용)
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['main_green_idx_list'] = []    # 각 교차로의 주 현시 인덱스 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['sub_green_idx_list'] = []     # 각 교차로의 나머지 현시 인덱스 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tl_idx_list'] = []            # 각 교차로의 tl_idx 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['remain_list'] = []            # 각 교차로의 잔여 녹색 시간 리스트(잔여 녹색 시간 = 주기 - 최소 녹색 시간의 합)
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['max_phase_list'] = []         # 각 교차로의 녹색 현시가 가장 긴 현시 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space_list'] = []      # 각 교차로의 action space 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_list_list'] = []       # 각 교차로의 녹색 시간 조정 action list(주 현시와 나머지 현시 조정)
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space_list'] = []       # 각 교차로의 state space 리스트
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list'] = []           # 각 교차로의 진입 link list(0-hop, 1-hop), 2차원으로 구분 없이 1차원으로 모든 link
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0'] = []         # 각 교차로의 진입 link list(0-hop), 2차원으로 구분 없이 1차원으로 모든 link
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1'] = []         # 각 교차로의 진입 link list(1-hop), 2차원으로 구분 없이 1차원으로 모든 link
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_list'] = []      # 각 교차로의 진입 link list(0-hop, 1-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0_list'] = []    # 각 교차로의 진입 link list(0-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1_list'] = []    # 각 교차로의 진입 link list(1-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list'] = []           # 각 교차로의 진입 lane list(0-hop, 1-hop), 2차원으로 구분 없이 1차원으로 모든 link
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0'] = []         # 각 교차로의 진입 lane list(0-hop), 2차원으로 구분 없이 1차원으로 모든 link
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1'] = []         # 각 교차로의 진입 lane list(1-hop), 2차원으로 구분 없이 1차원으로 모든 link
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_list'] = []      # 각 교차로의 진입 lane list(0-hop, 1-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0_list'] = []    # 각 교차로의 진입 lane list(0-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1_list'] = []    # 각 교차로의 진입 lane list(1-hop), 2차원으로 구분하여 각 교차로마다 구분 됨

        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['crossName_list'].append(target_tl_obj[tl_obj]['crossName'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tlid_list'].append(tl_obj)
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space'] += target_tl_obj[tl_obj]['state_space']
        if args.action=='gro':
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space'] += 2

            # todo should check correctness of value : 0..1,   .. (# of green phase  -1)
            # for offset
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'].append(0)
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'].append(target_tl_obj[tl_obj]['action_space'] - 1)

            # for green ratio
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'].append(0)
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'].append(target_tl_obj[tl_obj]['action_space'] - 1)

        elif args.action=='gt':
            num_controllable_green_signals = target_tl_obj[tl_obj]['action_space']
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space'] += num_controllable_green_signals
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'] += [-1.0] * num_controllable_green_signals
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'] += [+1.0] * num_controllable_green_signals

        else:
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space'] += 1
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'].append(0)
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'].append(target_tl_obj[tl_obj]['action_space'] - 1)

        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['offset_list'].append(target_tl_obj[tl_obj]['offset'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['minDur_list'].append(target_tl_obj[tl_obj]['minDur'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['maxDur_list'].append(target_tl_obj[tl_obj]['maxDur'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['cycle_list'].append(target_tl_obj[tl_obj]['cycle'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_list'].append(target_tl_obj[tl_obj]['duration'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['green_idx_list'].append(target_tl_obj[tl_obj]['green_idx'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_bins_list'].append(target_tl_obj[tl_obj]['duration_bins'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['main_green_idx_list'].append(target_tl_obj[tl_obj]['main_green_idx'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['sub_green_idx_list'].append(target_tl_obj[tl_obj]['sub_green_idx'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tl_idx_list'].append(target_tl_obj[tl_obj]['tl_idx'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['remain_list'].append(target_tl_obj[tl_obj]['remain'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['max_phase_list'].append(target_tl_obj[tl_obj]['max_phase'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space_list'].append(target_tl_obj[tl_obj]['action_space'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_list_list'].append(target_tl_obj[tl_obj]['action_list'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space_list'].append(target_tl_obj[tl_obj]['state_space'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list'] += target_tl_obj[tl_obj]['in_edge_list']
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0'] += target_tl_obj[tl_obj]['in_edge_list_0']
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1'] += target_tl_obj[tl_obj]['in_edge_list_1']
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_list'].append(target_tl_obj[tl_obj]['in_edge_list'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0_list'].append(target_tl_obj[tl_obj]['in_edge_list_0'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1_list'].append(target_tl_obj[tl_obj]['in_edge_list_1'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list'] += target_tl_obj[tl_obj]['in_lane_list']
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0'] += target_tl_obj[tl_obj]['in_lane_list_0']
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1'] += target_tl_obj[tl_obj]['in_lane_list_1']
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_list'].append(target_tl_obj[tl_obj]['in_lane_list'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0_list'].append(target_tl_obj[tl_obj]['in_lane_list_0'])
        sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1_list'].append(target_tl_obj[tl_obj]['in_lane_list_1'])

    if DBG_OPTIONS.PrintSaRelatedInfo:
        print("sa_obj")
        pprint.pprint(sa_obj, width=200, compact=True)

    return target_tl_obj, sa_obj, _lane_len


@deprecated(reason="use another function : gatherTsoOutputInfo")
def getAverageSpeedOfIntersection(tl_id, tl_obj, num_hop=0):
    '''
    get average speed of given intersection

    :param tl_id: inersection identifier
    :param tl_obj:  objects which holds TL information
    :param num_hop: number of hop to calculate speed
    :return:
    '''
    link_list = tl_obj[tl_id]['in_edge_list_0']

    if num_hop>0:
        link_list += tl_obj[tl_id]['in_edge_list_1']
    link_speed_list = []
    for link_id in link_list:
        link_speed_list.append(libsalt.link.getAverageSpeed(link_id))

    return np.average(link_speed_list)


@deprecated(reason="use another function : gatherTsoOutputInfo")
def getAverageTravelTimeOfIntersection(tl_id, tl_obj, num_hop=0):
    '''
    get average travel time of given intersection

    :param tl_id: inersection identifier
    :param tl_obj:  objects which holds TL information
    :param num_hop: number of hop to calculate speed
    :return:
    '''
    link_list = tl_obj[tl_id]['in_edge_list_0']

    if num_hop>0:
        link_list += tl_obj[tl_id]['in_edge_list_1']

    # todo check which one is suitable
    # todo check : performance improvement rate
    if DBG_OPTIONS.AVG_AVG: # avg_avg
        link_avg_time_list = []
        for link_id in link_list:
            sum_travel_time = libsalt.link.getSumTravelTime(link_id)
            sum_passed = libsalt.link.getSumPassed(link_id)
            avg_tt = 0.0
            if sum_passed > 0:
                avg_tt = sum_travel_time/sum_passed
            link_avg_time_list.append(avg_tt)

        avg_tt = np.average(link_avg_time_list)
    else: # sum_avg
        sum_travel_time = 0
        sum_passed = 0
        for link_id in link_list:
            sum_travel_time += libsalt.link.getSumTravelTime(link_id)
            sum_passed += libsalt.link.getSumPassed(link_id)
        avg_tt = 0.0
        if sum_passed > 0:
            avg_tt = sum_travel_time / sum_passed

    return avg_tt




@deprecated(reason="use another function : gatherTsoOutputInfo")
def getSumTravelTimeOfIntersection(tl_id, tl_obj, num_hop=0):
    '''
    get sum travel time of given intersection

    :param tl_id: inersection identifier
    :param tl_obj:  objects which holds TL information
    :param num_hop: number of hop to calculate speed
    :return:
    '''
    link_list = tl_obj[tl_id]['in_edge_list_0']

    if num_hop>0:
        link_list += tl_obj[tl_id]['in_edge_list_1']

    sum_travel_time = 0.0
    for link_id in link_list:
        sum_travel_time += libsalt.link.getSumTravelTime(link_id)

    return sum_travel_time





def initTsoOutputInfo():
    '''
    initialize dictionary to hold traffic signal optimization output
    '''
    info_dic = {}
    info_dic["avg_speed"] = []
    info_dic["avg_travel_time"] = []
    info_dic["sum_passed"] = []
    info_dic["sum_travel_time"] = []

    if DBG_OPTIONS.RichActionOutput:
        info_dic["offset"]=[]
        info_dic["duration"]=[]

    return info_dic



def appendTsoOutputInfo(info_dic, avg_speed, avg_tt, sum_passed, sum_travel_time):
    '''
    append statisitcal info to the dictionary for holding traffic signal optimization output
    :param info_dic: dic
    :param avg_speed:
    :param avg_tt:
    :param sum_passed:
    :param sum_travel_time:
    :return:
    '''
    info_dic["avg_speed"].append(avg_speed)
    info_dic["avg_travel_time"].append(avg_tt)
    info_dic["sum_passed"].append(sum_passed)
    info_dic["sum_travel_time"].append(sum_travel_time)
    return info_dic


def __convertDurationListIntoString(duration, separator):
    '''
     convert list into string
    :param duration: list, ex., [40, 3, 72, 3]
    :param separator:
    :retuen:  40_3_72_3 if separator is underscore(_)
    '''
    duration_str = str(duration)
    table = duration_str.maketrans({']':'',  # remove ]
                                    '[':'',  # remove [
                                    ' ':'',  # remove space
                                    ',':separator}) # convert comma into space
    duration_str = duration_str.translate(table)
    return duration_str


# if DBG_OPTIONS.RichActionOutput
def appendTsoOutputInfoSignal(info_dic, offset, duration):
    '''
    append traffic signal info to the dictionary for holding traffic signal optimization output
    :param info_dic: dic
    :param offset: int
    :param duration: list, ex., [18, 4, 72, 4, 18, 4, 28, 4, 25, 3]
    :retuen:
    '''
    info_dic["offset"].append(offset) # offset=144

    duration_str = __convertDurationListIntoString(duration, '_')

    info_dic["duration"].append(duration_str)

    return info_dic


def replaceTsoOutputInfo(info_dic, ith, avg_speed, avg_tt, sum_passed, sum_travel_time):
    '''
    append info to the dictionary for holding traffic signal optimization output
    :param info_dic: dic
    :param ith: index which indicates to replace
    :param avg_speed:
    :param avg_tt:
    :param sum_passed:
    :param sum_travel_time:
    :return:
    '''
    info_dic["avg_speed"][ith] = avg_speed
    info_dic["avg_travel_time"][ith] = avg_tt
    info_dic["sum_passed"][ith] = sum_passed
    info_dic["sum_travel_time"][ith]  = sum_travel_time
    return info_dic

def replaceTsoOutputInfoOffset(info_dic, ith, offset):
    info_dic["offset"][ith] = offset

    return info_dic


def replaceTsoOutputInfoDuration(info_dic, ith, duration):
    duration_str = __convertDurationListIntoString(duration, '_')
    info_dic["duration"][ith] = duration_str
    return info_dic

def replaceTsoOutputInfoSignal(info_dic, ith, offset, duration=[]):
    info_dic["offset"][ith] = offset

    if len(duration):
        duration_str = __convertDurationListIntoString(duration, '_')
        info_dic["duration"][ith] = duration_str

    return info_dic


def getTsoOutputInfo(info_dic, ith):
    '''
    append info to the dictionary for holding traffic signal optimization output
    :param info_dic: dic
    :param ith: index which indicates to get

    :return:
    '''
    avg_speed = info_dic["avg_speed"][ith]
    avg_tt = info_dic["avg_travel_time"][ith]
    sum_passed = info_dic["sum_passed"][ith]
    sum_travel_time = info_dic["sum_travel_time"][ith]
    return avg_speed, avg_tt, sum_passed, sum_travel_time

def getTsoOutputInfoSignal(info_dic, ith):
    offset = info_dic["offset"][ith]
    duration = info_dic["duration"][ith]
    return offset, duration


def gatherTsoOutputInfo(tl_id, tl_obj, num_hop=0):
    '''
    gather TSO-related information of given intersection

    TSO-related information : average speed, travel time, passed vehicle num

    :param tl_id: inersection identifier
    :param tl_obj:  objects which holds TL information
    :param num_hop: number of hop to calculate speed
    :return:
    '''
    link_list = tl_obj[tl_id]['in_edge_list_0']

    if num_hop>0:
        link_list += tl_obj[tl_id]['in_edge_list_1']
    link_speed_list = []
    link_avg_time_list = []
    sum_travel_time = 0
    sum_passed = 0
    for link_id in link_list:
        # average speed
        link_speed_list.append(libsalt.link.getAverageSpeed(link_id))

        # passed vehicle num
        passed = libsalt.link.getSumPassed(link_id)

        # travel time
        travel_time = libsalt.link.getSumTravelTime(link_id)
        if passed > 0:
            avg_tt = travel_time / passed
        else:
            avg_tt = 0

        link_avg_time_list.append(avg_tt)
        sum_passed += passed
        sum_travel_time += travel_time

    avg_speed = np.average(link_speed_list)
    avg_tt = np.average(link_avg_time_list)

    return avg_speed, avg_tt, sum_passed, sum_travel_time



def appendPhaseRewards(fn, sim_step, actions, reward_mgmt, sa_obj, sa_name_list, tl_obj, tl_id_list, tso_output_info_dic):
    '''
        write reward to given file
        this func is called in TEST-, SIMULATE-mode to write reward info which will be used by visualization tool

        :param fn: file name to store reward
        :param sim_step: simulation step
        :param actions: applied actions
        :param reward_mgmt: object for reward mgmt
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
    sa_reward_related_info_list = []
    if reward_mgmt.reward_unit == _REWARD_GATHER_UNIT_.SA:
        sa_reward_list = []
        for sa_idx in range(num_target):
            sa_reward = reward_mgmt.calculateSARewardInstantly(sa_idx, sim_step)
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

            # getTsoOutputInfo() & getTsoOutputInfoSignal() 합치자.
            if (sim_step % _RESULT_COMP_.SPEED_GATHER_INTERVAL) == 0:
                avg_speed, avg_tt, sum_passed, sum_travel_time = gatherTsoOutputInfo(tlid, tl_obj, num_hop=0)
                tso_output_info_dic = replaceTsoOutputInfo(tso_output_info_dic, i, avg_speed, avg_tt, sum_passed, sum_travel_time)
            else:
                avg_speed, avg_tt, sum_passed, sum_travel_time = getTsoOutputInfo(tso_output_info_dic, i)

            # getTsoOutputInfo() & getTsoOutputInfoSignal() 합치자.
            if DBG_OPTIONS.RichActionOutput:
                offset, duration = getTsoOutputInfoSignal(tso_output_info_dic, i)
                tl_action = f'{tl_action}#{offset}#{duration}'

            f.write("{},{},{},{},{},{},{},{},{}\n".format(sim_step,
                                                       tl_obj[tlid]['crossName'],
                                                       tl_action,
                                                       libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid),
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

            reward = reward_mgmt.calculateTLRewardInstantly(sa_idx, tlid, sim_step)

            tl_action = 0
            if len(actions) != 0:
                tl_idx = sa_obj[sa_name]['tlid_list'].index(tlid)
                tl_action = actions[sa_idx][tl_idx]

            if (sim_step % _RESULT_COMP_.SPEED_GATHER_INTERVAL) == 0:
                avg_speed, avg_tt, sum_passed, sum_travel_time = gatherTsoOutputInfo(tlid, tl_obj, num_hop=0)
                tso_output_info_dic = replaceTsoOutputInfo(tso_output_info_dic, i, avg_speed, avg_tt, sum_passed, sum_travel_time)
            else:
                avg_speed, avg_tt, sum_passed, sum_travel_time = getTsoOutputInfo(tso_output_info_dic, i)

            if DBG_OPTIONS.RichActionOutput:
                offset, duration = getTsoOutputInfoSignal(tso_output_info_dic, i)
                tl_action = f'{tl_action}#{offset}#{duration}'

            f.write("{},{},{},{},{},{},{},{},{}\n".format(sim_step,
                                                       tl_obj[tlid]['crossName'],
                                                       tl_action,
                                                       libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid),
                                                       reward,
                                                       avg_speed,
                                                       avg_tt,
                                                       sum_passed,
                                                       sum_travel_time))

    f.close()



def startTimeConvert(f_path, f_name, start_hour):
    '''
    convert start time

    :param f_path: route file path
    :param f_name: route file name
    :param start_hour:
    :return:
    '''
    # import xml.etree.ElementTree as ET

    start_time_second = float(start_hour * 60 * 60)

    print("start_time_second={}".format(start_time_second))
    tree = ET.parse(f"{f_path}/{f_name}")
    root = tree.getroot()
    vehicles = root.findall("vehicle")

    for x in vehicles:
        x.attrib["depart"] = str(float(x.attrib["depart"]) + start_time_second)

    cvted_file_name = "cvted_"+f_name
    tree.write(f"{f_path}/{cvted_file_name}")



def testStartTimeConvert():

    file_path = "/tmp/routes"

    in_file_dic = {7 : "Doan_traffic_07-09_KAIST_2022.rou.xml",
                   9 : "Doan_traffic_09-11_KAIST_2022.rou.xml",
                   14: "Doan_traffic_14-16_KAIST_2022.rou.xml",
                   17:  "Doan_traffic_17-19_KAIST_2022.rou.xml",
                   20: "Doan_traffic_20-22_KAIST_2022.rou.xml",
                   23 : "Doan_traffic_23-01_KAIST_2022.rou.xml" }

    for start_hour in in_file_dic.keys():
        print(start_hour, in_file_dic[start_hour])
        startTimeConvert(file_path, in_file_dic[start_hour], int(start_hour))
