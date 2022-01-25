import numpy as np
from xml.etree.ElementTree import parse
import sys
import os

from config import TRAIN_CONFIG
print(TRAIN_CONFIG)
sys.path.append(TRAIN_CONFIG['libsalt_dir'])

IS_DOCKERIZE = TRAIN_CONFIG['IS_DOCKERIZE']

import libsalt

def get_objs(args, trafficSignal, targetList_input2, edge_file_path, salt_scenario, startStep):
    target_tl_obj = {}
    sa_obj = {}

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

            target_tl_obj[x.attrib['nodeID']]['offset'] = int(x.find('schedule').attrib['offset'])
            target_tl_obj[x.attrib['nodeID']]['minDur'] = [int(y.attrib['minDur']) if 'minDur' in y.attrib else int(y.attrib['duration']) for
                                                                y in x.findall("schedule/phase")]
            target_tl_obj[x.attrib['nodeID']]['maxDur'] = [int(y.attrib['maxDur']) if 'maxDur' in y.attrib else int(y.attrib['duration']) for
                                                                y in x.findall("schedule/phase")]
            target_tl_obj[x.attrib['nodeID']]['cycle'] = np.sum([int(y.attrib['duration']) for y in x.findall("schedule/phase")])
            target_tl_obj[x.attrib['nodeID']]['duration'] = [int(y.attrib['duration']) for y in x.findall("schedule/phase")]
            tmp_duration_list = np.array([int(y.attrib['duration']) for y in x.findall("schedule/phase")])
            # target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(tmp_duration_list > 5)
            target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(np.array(target_tl_obj[x.attrib['nodeID']]['minDur']) != np.array(target_tl_obj[x.attrib['nodeID']]['maxDur']))

            ### for select discrete action with the current phase ratio from tanH Prob.
            dur_arr = []
            for g in target_tl_obj[x.attrib['nodeID']]['green_idx'][0]:
                dur_arr.append(target_tl_obj[x.attrib['nodeID']]['duration'][g])
            dur_ratio = dur_arr / np.sum(dur_arr)
            dur_ratio
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
            phase_numbers.append(len(target_tl_obj[x.attrib['nodeID']]['green_idx'][0]))
            i += 1

    max_phase_length = int(np.max(phase_numbers))


    if IS_DOCKERIZE:
        tree = parse(edge_file_path)
    else:
        tree = parse(os.getcwd() + '/data/envs/salt/doan/doan_20210401.edg.xml')

    root = tree.getroot()

    edge = root.findall("edge")

    target_tl_id_list = list(target_tl_obj.keys())

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

    max_edge_length = int(np.max(_edge_len))
    print(target_tl_obj)
    print(max_edge_length)

    done = False

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
            target_tl_obj[target]['state_space'] = len(_lane_list) * 2 + 1
        else:
            target_tl_obj[target]['state_space'] = len(_lane_list) + 1
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

