import numpy as np
from xml.etree.ElementTree import parse
import sys
import os

from config import TRAIN_CONFIG
sys.path.append(TRAIN_CONFIG['libsalt_dir'])

import libsalt

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

def get_objs(args, trafficSignal, targetList_input2, edge_file_path, salt_scenario, startStep):
    target_tl_obj = {}

    phase_numbers = []
    i=0
    for x in trafficSignal:
        if x.attrib['signalGroup'] in targetList_input2:
            target_tl_obj[x.attrib['nodeID']] = {}
            target_tl_obj[x.attrib['nodeID']]['crossName'] = x.attrib['crossName']

            _signalGroup = x.attrib['signalGroup']
            if "SA" not in _signalGroup:
                _signalGroup = "SA " + _signalGroup
            if "SA " not in _signalGroup:
                _signalGroup = _signalGroup.replace("SA", "SA ")

            target_tl_obj[x.attrib['nodeID']]['signalGroup'] = _signalGroup
            if _signalGroup == "SA 1":
                s_id = '11'
            elif _signalGroup == "SA 56" or _signalGroup == "SA 111":
                s_id = '5'
            elif _signalGroup == "SA 107":
                s_id = '1'
            else:
                s_id = '2'

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
            phase_numbers.append(len(target_tl_obj[x.attrib['nodeID']]['green_idx'][0]))
            i += 1

    tree = parse(edge_file_path)
    root = tree.getroot()
    edge = root.findall("edge")

    target_tl_id_list = list(target_tl_obj.keys())

    near_tl_obj = {}
    for i in target_tl_id_list:
        near_tl_obj[i] = {}
        near_tl_obj[i]['in_edge_list'] = []
        near_tl_obj[i]['in_edge_list_0'] = []
        near_tl_obj[i]['in_edge_list_1'] = []

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

    print(target_tl_obj)

    libsalt.start(salt_scenario)
    libsalt.setCurrentStep(startStep)

    print("init", [libsalt.trafficsignal.getTLSConnectedLinkID(x) for x in target_tl_id_list])
    print("init", [libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(x) for x in target_tl_id_list])
    print("init", [libsalt.trafficsignal.getLastTLSPhaseSwitchingTimeByNodeID(x) for x in target_tl_id_list])
    print("init", [len(libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(x).myPhaseVector) for x in target_tl_id_list])
    print("init", [libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(x).myPhaseVector[0][1] for x in target_tl_id_list])

    _lane_len = []
    for target in target_tl_obj:
        _lane_list = []
        _lane_list_0 = []
        for edge in target_tl_obj[target]['in_edge_list_0']:
            for lane in range(libsalt.link.getNumLane(edge)):
                _lane_id = "{}_{}".format(edge, lane)
                _lane_list.append(_lane_id)
                _lane_list_0.append((_lane_id))
        target_tl_obj[target]['in_lane_list_0'] = _lane_list_0
        _lane_list_1 = []
        for edge in target_tl_obj[target]['in_edge_list_1']:
            for lane in range(libsalt.link.getNumLane(edge)):
                _lane_id = "{}_{}".format(edge, lane)
                _lane_list.append(_lane_id)
                _lane_list_1.append((_lane_id))
        target_tl_obj[target]['in_lane_list_1'] = _lane_list_1
        target_tl_obj[target]['in_lane_list'] = _lane_list
        if args.state == 'vd':
            target_tl_obj[target]['state_space'] = len(_lane_list_0) * 2 + 1
        else:
            target_tl_obj[target]['state_space'] = len(_lane_list_0) + 1
        _lane_len.append(len(_lane_list))

    libsalt.close()
    ### for SAPPO Agent ###
    sa_obj = {}
    for tl_obj in target_tl_obj:
        if target_tl_obj[tl_obj]['signalGroup'] not in sa_obj:
            sa_obj[target_tl_obj[tl_obj]['signalGroup']] = {}
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['crossName_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tlid_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space'] = 0
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space'] = 0
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['offset_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['minDur_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['maxDur_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['cycle_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['green_idx_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_bins_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['main_green_idx_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['sub_green_idx_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tl_idx_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['remain_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['max_phase_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_list_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0_list'] = []
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1_list'] = []

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
        
    return target_tl_obj, sa_obj, _lane_len

