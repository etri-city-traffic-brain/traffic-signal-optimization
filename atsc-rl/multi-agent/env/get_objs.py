import numpy as np
from xml.etree.ElementTree import parse
import sys
import os
import pprint

from config import TRAIN_CONFIG
sys.path.append(TRAIN_CONFIG['libsalt_dir'])

import libsalt

### 녹색 시간 조정 actioon 생성
def getActionList(phase_num, max_phase):
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
        #     action_list = [x.tolist() for x in meshgrid if x[max_phase]!=0 and x[max_phase]+np.sum(x[mask])==0 and np.min(np.abs(x[mask]))==0 and np.max(np.abs(x[mask]))==1]

        if phase_num == 1:
            action_list = [[0]]
        else:
            action_list = [x.tolist() for x in meshgrid if
                           x[max_phase] != 0 and x[max_phase] + np.sum(x[mask]) == 0 and np.min(
                               np.abs(x[mask])) == 0 and np.max(np.abs(x[mask])) == 1 and x[max_phase] != np.min(
                               x[mask]) and x[max_phase] != np.max(x[mask])]
    else:
        meshgrid = np.array(np.meshgrid(_pos, _pos, _pos, _pos, _pos, _pos)).T.reshape(-1, phase_num)
        action_list = [x.tolist() for x in meshgrid if
                       x[max_phase] != 0 and x[max_phase] + np.sum(x[mask]) == 0 and np.min(
                           np.abs(x[mask])) == 0 and np.max(np.abs(x[mask])) == 1 and x[max_phase] != np.min(
                           x[mask]) - 1 and x[max_phase] != np.max(x[mask]) + 1 and x[max_phase] != np.min(x[mask]) and
                       x[max_phase] != np.max(x[mask])]

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

def getActionList_v2(phase_num, max_phase):
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

        #     action_list = [x.tolist() for x in meshgrid if x[max_phase]!=0 and x[max_phase]+np.sum(x[mask])==0 and np.min(np.abs(x[mask]))==0 and np.max(np.abs(x[mask]))==1]
        action_list = [x.tolist() for x in meshgrid if
                       x[max_phase] != 0 and x[max_phase] + np.sum(x[mask]) == 0 and np.min(
                           np.abs(x[mask])) == 0 and np.max(np.abs(x[mask])) == 1 and x[max_phase] != np.min(
                           x[mask]) and x[max_phase] != np.max(x[mask]) and x[max_phase]>0]
    else:
        meshgrid = np.array(np.meshgrid(_pos, _pos, _pos, _pos, _pos, _pos)).T.reshape(-1, phase_num)
        action_list = [x.tolist() for x in meshgrid if
                       x[max_phase] != 0 and x[max_phase] + np.sum(x[mask]) == 0 and np.min(
                           np.abs(x[mask])) == 0 and np.max(np.abs(x[mask])) == 1 and x[max_phase] != np.min(
                           x[mask]) - 1 and x[max_phase] != np.max(x[mask]) + 1 and x[max_phase] != np.min(x[mask]) and
                       x[max_phase] != np.max(x[mask]) and x[max_phase]>0]

#     tmp = list([1] * phase_num)
#     tmp[max_phase] = -(phase_num - 1)
#     action_list.append(tmp)

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
def getPossibleActionList(args, duration, minDur, maxDur, green_idx, actionList):

    duration = np.array(duration)
    minGreen = np.array(minDur)
    maxGreen = np.array(maxDur)
    green_idx = np.array(green_idx)

    new_actionList = []

    for action in actionList:
        npsum = 0
        newDur = duration[green_idx] + np.array(action) * args.addTime
        npsum += np.sum(minGreen[green_idx] > newDur)
        npsum += np.sum(maxGreen[green_idx] < newDur)
        if npsum == 0:
            new_actionList.append(action)
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
        #offset = traffic_signal.find("TODPlan").attrib['offset']
        schedule = traffic_signal.find("TODPlan").attrib['defaultPlan']
        #start_time = given_start_time
    else:
        #offset = all_plan[idx].attrib['offset']
        schedule = all_plan[idx].attrib['schedule']
        #start_time = all_plan[idx].attrib['startTime']

    # if 0:
    #     print("given_start_time={} offset={} schedule={} startTime={}".format(given_start_time, offset, schedule, start_time))
    return schedule
###--------- end of addition

def constructTSSRelatedInfo(args, trafficSignal, targetList_input2):
    '''
    construce TSS related info from given traffic environment
    # :param tss_file_path: file path of TSS
    :param trafficSignal: parse(file path of TSS).getroot().findall("TrafficSignal")
    :param targetList_input2: target signal group info
    :return:  an object which contains TSS related info
    '''
    # tree = parse(tss_file_path)
    # root = tree.getroot()
    # trafficSignal = root.findall("trafficSignal")

    target_tl_obj = {}
    i = 0
    for x in trafficSignal:
        sg = x.attrib['signalGroup'].strip()  # add by hunsooni
        if sg in targetList_input2:
            target_tl_obj[x.attrib['nodeID']] = {}
            target_tl_obj[x.attrib['nodeID']]['crossName'] = x.attrib['crossName']

            _signalGroup = x.attrib['signalGroup']
            if "SA" not in _signalGroup:
                _signalGroup = "SA " + _signalGroup
            if "SA " not in _signalGroup:
                _signalGroup = _signalGroup.replace("SA", "SA ")

            target_tl_obj[x.attrib['nodeID']]['signalGroup'] = _signalGroup

            # hunsooni : 범용으로 바꿔봄. start_time 이 시나리오 참고하여 제대로 설정되어야 함.
            # ----------- begin of modification : by hunsooni
            if 0:
                if _signalGroup == "SA 1":
                    s_id = '11'
                elif _signalGroup == "SA 56" or _signalGroup == "SA 111":
                    s_id = '5'
                elif _signalGroup == "SA 107":
                    s_id = '1'
                else:
                    s_id = '2'
            else:
                s_id = getScheduleID(x, args.start_time)
            # ----------- end of modification : by hunsooni

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
    :param salt_scenario: scenario file path
    :param target_tl_obj: an object to store constructed LANE related info
    :return:
    '''
    startStep = 0

    libsalt.start(salt_scenario)
    libsalt.setCurrentStep(startStep)

    # print("init", [libsalt.trafficsignal.getTLSConnectedLinkID(x) for x in target_tl_id_list])
    # print("init", [libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(x) for x in target_tl_id_list])
    # print("init", [libsalt.trafficsignal.getLastTLSPhaseSwitchingTimeByNodeID(x) for x in target_tl_id_list])
    # print("init", [len(libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(x).myPhaseVector) for x in target_tl_id_list])
    # print("init", [libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(x).myPhaseVector[0][1] for x in target_tl_id_list])

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
    # print(target_tl_obj)

    libsalt.close()

    return target_tl_obj, _lane_len

### 신호 최적화 대상 교차로 및 교차로 그룹에 대한 정보를 object로 생성
def get_objs(args, trafficSignal, targetList_input2, edge_file_path, salt_scenario, startStep):
    target_tl_obj = constructTSSRelatedInfo(args, trafficSignal, targetList_input2)

    ## get the identifier of target intersection to optimize signal
    target_tl_id_list = list(target_tl_obj.keys())

    ## get EDGE info which are belong to the target intersection group for optimizing signal
    target_tl_obj = constructEdgeRelatedInfo(edge_file_path, target_tl_id_list, target_tl_obj)

    ## build incomming LANE related info by executing the simulator
    target_tl_obj, _lane_len = constructLaneRelatedInfo(args, salt_scenario, target_tl_obj)

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
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space'] = 0    # action space - 교차로 그룹에 속한 교차로 수
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

    print("sa_obj")
    pprint.pprint(sa_obj, width=200, compact=True)
    return target_tl_obj, sa_obj, _lane_len

