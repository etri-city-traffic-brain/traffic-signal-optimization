import argparse

import numpy as np
import matplotlib.pyplot as plt
import time

from policy.ddqn import DDQN
from policy.ppo import PPOAgent

from config import TRAIN_CONFIG

IS_DOCKERIZE = TRAIN_CONFIG['IS_DOCKERIZE']

if IS_DOCKERIZE:
    from env.salt_PennStateAction import SALT_doan_multi_PSA_test, getScenarioRelatedFilePath, getScenarioRelatedBeginEndTime
    from env.sappo_noConst import SALT_SAPPO_noConst, getScenarioRelatedFilePath, getScenarioRelatedBeginEndTime
else:
    from env.salt_PennStateAction import SALT_doan_multi_PSA
    from env.sappo_noConst import SALT_SAPPO_noConst
    from env.sappo_offset import SALT_SAPPO_offset
    from env.sappo_offset_single import SALT_SAPPO_offset_single


from config import TRAIN_CONFIG

import sys
import os
import pandas as pd

import subprocess

from xml.etree.ElementTree import parse

from config import TRAIN_CONFIG

sys.path.append(TRAIN_CONFIG['libsalt_dir'])

import libsalt


addTime = 2
state_weight = 1



def result_comp(args, ft_output, rl_output, model_num):
    if IS_DOCKERIZE:
        if 0:
            scenario_file_path, _, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args)
            tree = parse(tss_file_path)
        else:
            scenario_file_path, node_file_path, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args)
            edge_file_path = "magic/doan_20210401.edg.xml"
            tss_file_path = "magic/doan(without dan).tss.xml"

        tree = parse(tss_file_path)
    else:
        tree = parse(os.getcwd() + '/data/envs/salt/doan/doan_20211207.tss.xml')


    root = tree.getroot()

    trafficSignal = root.findall("trafficSignal")

    target_tl_obj = {}
    phase_numbers = []
    i = 0

    targetList_input = args.target_TL.split(',')

    targetList_input2 = []

    for tl_i in targetList_input:
        targetList_input2.append(tl_i)  ## ex. SA 101
        targetList_input2.append(tl_i.split(" ")[1])  ## ex.101
        targetList_input2.append(tl_i.replace(" ", ""))  ## ex. SA101

    for x in trafficSignal:
        if x.attrib['signalGroup'] in targetList_input2:
            target_tl_obj[x.attrib['nodeID']] = {}
            target_tl_obj[x.attrib['nodeID']]['crossName'] = x.attrib['crossName']
            target_tl_obj[x.attrib['nodeID']]['signalGroup'] = x.attrib['signalGroup']
            target_tl_obj[x.attrib['nodeID']]['offset'] = int(x.find('schedule').attrib['offset'])
            target_tl_obj[x.attrib['nodeID']]['minDur'] = [
                int(y.attrib['minDur']) if 'minDur' in y.attrib else int(y.attrib['duration']) for
                y in x.findall("schedule/phase")]
            target_tl_obj[x.attrib['nodeID']]['maxDur'] = [
                int(y.attrib['maxDur']) if 'maxDur' in y.attrib else int(y.attrib['duration']) for
                y in x.findall("schedule/phase")]
            target_tl_obj[x.attrib['nodeID']]['cycle'] = np.sum(
                [int(y.attrib['duration']) for y in x.findall("schedule/phase")])
            target_tl_obj[x.attrib['nodeID']]['duration'] = [int(y.attrib['duration']) for y in
                                                             x.findall("schedule/phase")]
            tmp_duration_list = np.array([int(y.attrib['duration']) for y in x.findall("schedule/phase")])
            target_tl_obj[x.attrib['nodeID']]['green_idx'] = np.where(tmp_duration_list > 5)
            target_tl_obj[x.attrib['nodeID']]['main_green_idx'] = np.where(
                tmp_duration_list == np.max(tmp_duration_list))
            target_tl_obj[x.attrib['nodeID']]['sub_green_idx'] = list(set(np.where(tmp_duration_list > 5)[0]) - set(
                np.where(tmp_duration_list == np.max(tmp_duration_list))[0]))
            target_tl_obj[x.attrib['nodeID']]['tl_idx'] = i
            target_tl_obj[x.attrib['nodeID']]['remain'] = target_tl_obj[x.attrib['nodeID']]['cycle'] - np.sum(
                target_tl_obj[x.attrib['nodeID']]['minDur'])
            target_tl_obj[x.attrib['nodeID']]['action_space'] = (len(
                target_tl_obj[x.attrib['nodeID']]['green_idx'][0]) - 1) * 2

            phase_numbers.append(len(target_tl_obj[x.attrib['nodeID']]['green_idx'][0]))
            i += 1

    target_tl_id_list = list(target_tl_obj.keys())
    agent_num = len(target_tl_id_list)

    if IS_DOCKERIZE:
        salt_scenario = scenario_file_path
        tree = parse(edge_file_path)
    else:
        salt_scenario = 'data/envs/salt/doan/doan_20211207.scenario.json'
        tree = parse(os.getcwd() + '/data/envs/salt/doan/doan_20211207.edg.xml')

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

    startStep = 0

    done = False

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
        target_tl_obj[target]['state_space'] = len(_lane_list)
        _lane_len.append(len(_lane_list))
    # print(target_tl_obj)

    libsalt.close()

    simulationSteps = 0

    print("\nstate_weight {} addtime {} model_num {}".format(state_weight, addTime, model_num))
    print("target_tl_obj", target_tl_obj)
    total_output = pd.DataFrame()

    for tl in target_tl_obj:
        if "SA " not in target_tl_obj[tl]['signalGroup']:
            target_tl_obj[tl]['signalGroup'] = 'SA ' + target_tl_obj[tl]['signalGroup']

        individual_output = pd.DataFrame(
            {'name': [target_tl_obj[tl]['crossName']], 'SA': [target_tl_obj[tl]['signalGroup']]})

        in_edge_list = []
        in_edge_list_0 = []
        in_edge_list = np.append(in_edge_list, target_tl_obj[tl]['in_edge_list'])
        in_edge_list_0 = np.append(in_edge_list_0, target_tl_obj[tl]['in_edge_list_0'])
        print(target_tl_obj[tl]['crossName'], target_tl_obj[tl]['in_edge_list_0'])
        # print(target_tl_obj[tl]['in_edge_list_0'])
        ft_output2 = ft_output[ft_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list_0) + '$', na=False)]
        rl_output2 = rl_output[rl_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list_0) + '$', na=False)]
        ft_output3 = ft_output[ft_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list) + '$', na=False)]
        rl_output3 = rl_output[rl_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list) + '$', na=False)]
        ft_output2 = ft_output2[ft_output2['intervalbegin'] >= 3600]
        rl_output2 = rl_output2[rl_output2['intervalbegin'] >= 3600]
        ft_output3 = ft_output3[ft_output3['intervalbegin'] >= 3600]
        rl_output3 = rl_output3[rl_output3['intervalbegin'] >= 3600]

        varList = ['VehPassed', 'AverageSpeed', 'WaitingTime', 'AverageDensity', 'SumTravelTime', 'WaitingQLength']
        varOp = [1, 1, -1, -1, -1, -1]
        varOp2 = ['sum', 'mean', 'sum', 'mean', 'sum', 'mean']

        for v in range(len(varList)):
            if varOp2[v] == 'sum':
                ft_passed = np.sum(ft_output2[varList[v]])
                rl_passed = np.sum(rl_output2[varList[v]])
                imp = varOp[v] * (rl_passed - ft_passed) / ft_passed * 100
                ft_passed = np.round(ft_passed, 2)
                rl_passed = np.round(rl_passed, 2)
                imp = np.round(imp, 2)

                print("0-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(varList[v], varOp2[v], ft_passed,
                                                                                  varList[v], varOp2[v], rl_passed,
                                                                                  imp))
                individual_output = pd.concat(
                    [individual_output, pd.DataFrame({'ft_{}_{}_0hop'.format(varList[v], varOp2[v]): [ft_passed],
                                                      'rl_{}_{}_0hop'.format(varList[v], varOp2[v]): [rl_passed],
                                                      'imp_{}_{}_0hop'.format(varList[v], varOp2[v]): [imp]})], axis=1)

                ft_passed = np.sum(ft_output3[varList[v]])
                rl_passed = np.sum(rl_output3[varList[v]])
                ft_passed = np.round(ft_passed, 2)
                rl_passed = np.round(rl_passed, 2)
                imp = varOp[v] * (rl_passed - ft_passed) / ft_passed * 100
                imp = np.round(imp, 2)

                print("1-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(varList[v], varOp2[v], ft_passed,
                                                                                  varList[v], varOp2[v], rl_passed,
                                                                                  imp))
                individual_output = pd.concat(
                    [individual_output, pd.DataFrame({'ft_{}_{}_1hop'.format(varList[v], varOp2[v]): [ft_passed],
                                                      'rl_{}_{}_1hop'.format(varList[v], varOp2[v]): [rl_passed],
                                                      'imp_{}_{}_1hop'.format(varList[v], varOp2[v]): [imp]})], axis=1)

            else:
                ft_passed = np.mean(ft_output2[varList[v]])
                rl_passed = np.mean(rl_output2[varList[v]])
                ft_passed = np.round(ft_passed, 2)
                rl_passed = np.round(rl_passed, 2)
                imp = varOp[v] * (rl_passed - ft_passed) / ft_passed * 100
                imp = np.round(imp, 2)
                print("0-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(varList[v], varOp2[v], ft_passed,
                                                                                  varList[v], varOp2[v], rl_passed,
                                                                                  imp))
                individual_output = pd.concat(
                    [individual_output, pd.DataFrame({'ft_{}_{}_0hop'.format(varList[v], varOp2[v]): [ft_passed],
                                                      'rl_{}_{}_0hop'.format(varList[v], varOp2[v]): [rl_passed],
                                                      'imp_{}_{}_0hop'.format(varList[v], varOp2[v]): [imp]})], axis=1)

                ft_passed = np.mean(ft_output3[varList[v]])
                rl_passed = np.mean(rl_output3[varList[v]])
                ft_passed = np.round(ft_passed, 2)
                rl_passed = np.round(rl_passed, 2)
                imp = varOp[v] * (rl_passed - ft_passed) / ft_passed * 100
                imp = np.round(imp, 2)

                print("1-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(varList[v], varOp2[v], ft_passed,
                                                                                  varList[v], varOp2[v], rl_passed,
                                                                                  imp))
                individual_output = pd.concat(
                    [individual_output, pd.DataFrame({'ft_{}_{}_1hop'.format(varList[v], varOp2[v]): [ft_passed],
                                                      'rl_{}_{}_1hop'.format(varList[v], varOp2[v]): [rl_passed],
                                                      'imp_{}_{}_1hop'.format(varList[v], varOp2[v]): [imp]})], axis=1)

        total_output = pd.concat([total_output, individual_output])

    in_edge_list = []
    in_edge_list_0 = []

    individual_output = pd.DataFrame({'name': ['total']})
    for tl in target_tl_obj:
        in_edge_list = np.append(in_edge_list, target_tl_obj[tl]['in_edge_list'])
        in_edge_list_0 = np.append(in_edge_list_0, target_tl_obj[tl]['in_edge_list_0'])
        print(target_tl_obj[tl]['crossName'], target_tl_obj[tl]['in_edge_list_0'])
        # print(target_tl_obj[tl]['in_edge_list_0'])
    ft_output2 = ft_output[ft_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list_0) + '$', na=False)]
    rl_output2 = rl_output[rl_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list_0) + '$', na=False)]
    ft_output3 = ft_output[ft_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list) + '$', na=False)]
    rl_output3 = rl_output[rl_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list) + '$', na=False)]
    ft_output2 = ft_output2[ft_output2['intervalbegin'] >= 3600]
    rl_output2 = rl_output2[rl_output2['intervalbegin'] >= 3600]
    ft_output3 = ft_output3[ft_output3['intervalbegin'] >= 3600]
    rl_output3 = rl_output3[rl_output3['intervalbegin'] >= 3600]

    for v in range(len(varList)):
        if varOp2[v] == 'sum':
            ft_passed = np.sum(ft_output2[varList[v]])
            rl_passed = np.sum(rl_output2[varList[v]])
            imp = varOp[v] * (rl_passed - ft_passed) / ft_passed * 100
            ft_passed = np.round(ft_passed, 2)
            rl_passed = np.round(rl_passed, 2)
            imp = np.round(imp, 2)

            print("0-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(varList[v], varOp2[v], ft_passed,
                                                                              varList[v], varOp2[v], rl_passed, imp))
            individual_output = pd.concat(
                [individual_output, pd.DataFrame({'ft_{}_{}_0hop'.format(varList[v], varOp2[v]): [ft_passed],
                                                  'rl_{}_{}_0hop'.format(varList[v], varOp2[v]): [rl_passed],
                                                  'imp_{}_{}_0hop'.format(varList[v], varOp2[v]): [imp]})], axis=1)

            ft_passed = np.sum(ft_output3[varList[v]])
            rl_passed = np.sum(rl_output3[varList[v]])
            ft_passed = np.round(ft_passed, 2)
            rl_passed = np.round(rl_passed, 2)
            imp = varOp[v] * (rl_passed - ft_passed) / ft_passed * 100
            imp = np.round(imp, 2)

            print("1-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(varList[v], varOp2[v], ft_passed,
                                                                              varList[v], varOp2[v], rl_passed, imp))
            individual_output = pd.concat(
                [individual_output, pd.DataFrame({'ft_{}_{}_1hop'.format(varList[v], varOp2[v]): [ft_passed],
                                                  'rl_{}_{}_1hop'.format(varList[v], varOp2[v]): [rl_passed],
                                                  'imp_{}_{}_1hop'.format(varList[v], varOp2[v]): [imp]})], axis=1)

        else:
            ft_passed = np.mean(ft_output2[varList[v]])
            rl_passed = np.mean(rl_output2[varList[v]])
            ft_passed = np.round(ft_passed, 2)
            rl_passed = np.round(rl_passed, 2)
            imp = varOp[v] * (rl_passed - ft_passed) / ft_passed * 100
            imp = np.round(imp, 2)
            print("0-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(varList[v], varOp2[v], ft_passed,
                                                                              varList[v], varOp2[v], rl_passed, imp))
            individual_output = pd.concat(
                [individual_output, pd.DataFrame({'ft_{}_{}_0hop'.format(varList[v], varOp2[v]): [ft_passed],
                                                  'rl_{}_{}_0hop'.format(varList[v], varOp2[v]): [rl_passed],
                                                  'imp_{}_{}_0hop'.format(varList[v], varOp2[v]): [imp]})], axis=1)

            ft_passed = np.mean(ft_output3[varList[v]])
            rl_passed = np.mean(rl_output3[varList[v]])
            ft_passed = np.round(ft_passed, 2)
            rl_passed = np.round(rl_passed, 2)
            imp = varOp[v] * (rl_passed - ft_passed) / ft_passed * 100
            imp = np.round(imp, 2)

            print("1-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(varList[v], varOp2[v], ft_passed,
                                                                              varList[v], varOp2[v], rl_passed, imp))
            individual_output = pd.concat(
                [individual_output, pd.DataFrame({'ft_{}_{}_1hop'.format(varList[v], varOp2[v]): [ft_passed],
                                                  'rl_{}_{}_1hop'.format(varList[v], varOp2[v]): [rl_passed],
                                                  'imp_{}_{}_1hop'.format(varList[v], varOp2[v]): [imp]})], axis=1)

    total_output = pd.concat([total_output, individual_output])

    total_output = total_output.sort_values(by=["SA"], ascending=True)

    return total_output


def ft_simulate(args):
    if IS_DOCKERIZE:
        salt_scenario = args.scenario_file_path
    else:
        salt_scenario = 'data/envs/salt/doan/doan_20211207_ft.scenario.json'

    if IS_DOCKERIZE:
        if 0:
            start_time = args.start_time
            trial_len = args.end_time - args.start_time
        else:
            scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
            start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
            end_time = args.end_time if args.end_time < scenario_end else scenario_end

            trial_len = end_time - start_time
    else:
        start_time = args.testStartTime
        trial_len = args.testEndTime - args.testStartTime

    env = SALT_doan_multi_PSA(args)

    for target_tl in list(env.target_tl_obj.keys()):
        env.target_tl_obj[target_tl]['crossName']

    if IS_DOCKERIZE:
        output_ft_dir = '{}/output/ft'.format(args.io_home)
        fn_ft_phase_reward_output = "{}/ft_phase_reward_output.txt".format(output_ft_dir)

        f = open(fn_ft_phase_reward_output, mode='w+', buffering=-1, encoding='utf-8', errors=None, newline=None,
                                  closefd=True, opener=None)
    else:
        f = open("output/ft/ft_phase_reward_output.txt", mode='w+', buffering=-1, encoding='utf-8', errors=None,
                 newline=None,
                 closefd=True, opener=None)

    f.write('step,tl_name,actions,phase,reward\n')
    f.close()

    libsalt.start(salt_scenario)
    libsalt.setCurrentStep(start_time)

    for i in range(trial_len):
        libsalt.simulationStep()
        for target_tl in list(env.target_tl_obj.keys()):
            env.target_tl_obj[target_tl]['crossName']

        if IS_DOCKERIZE:
            f = open(fn_ft_phase_reward_output, mode='a+', buffering=-1, encoding='utf-8', errors=None,
                    newline=None,
                    closefd=True, opener=None)
        else:
            f = open("output/ft/ft_phase_reward_output.txt", mode='a+', buffering=-1, encoding='utf-8', errors=None,
                     newline=None,
                     closefd=True, opener=None)

        for target_tl in list(env.target_tl_obj.keys()):
            tlid = target_tl
            f.write("{},{},{},{},{}\n".format(libsalt.getCurrentStep(), env.target_tl_obj[target_tl]['crossName'], 0,
                                              libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid), 0))
        f.close()

    print("ft_step {}".format(libsalt.getCurrentStep()))
    libsalt.close()


def ddqn_test(args, trial, problem_var):
    model_num = trial

    if IS_DOCKERIZE:
        salt_scenario = args.scenario_file_path
    else:
        salt_scenario = 'data/envs/salt/doan/doan_2021_ft.scenario.json'


    # cmd = 'sudo ../traffic-simulator/bin/./salt-standalone data/envs/salt/doan/doan_2021_ft.scenario.json'
    # so = os.popen(cmd).read()
    # print(so)

    if IS_DOCKERIZE:
        if 0:
            start_time = args.start_time
            trial_len = args.end_time - args.start_time
        else:
            scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
            start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
            end_time = args.end_time if args.end_time < scenario_end else scenario_end

            trial_len = end_time - start_time

    else:
        start_time = args.testStartTime
        trial_len = args.testEndTime - args.testStartTime

    if IS_DOCKERIZE:
        if args.result_comp:
            print("Start fixed time scenario for the result compare")
            ft_simulate(args)
            print("End fixed time scenario")
    else:
        if args.resultComp:
            print("Start fixed time scenario for the result compare")
            # ft_simulate(args, trial, problem_var)
            ft_simulate(args)
            print("End fixed time scenario")

    problem = "SALT_doan_multi_PSA_test"
    env = SALT_doan_multi_PSA_test(args)

    agent_num = env.agent_num
    action_mask = env.action_mask

    # updateTargetNetwork = 1000
    dqn_agent = []
    state_space_arr = []
    for i in range(agent_num):
        target_tl = list(env.target_tl_obj.keys())[i]
        state_space = env.target_tl_obj[target_tl]['state_space']
        state_space_arr.append(state_space)
        action_space = env.target_tl_obj[target_tl]['action_space']
        dqn_agent.append(DDQN(args=args, env=env, state_space=state_space, action_space=action_space, epsilon=0, epsilon_min=0))

        if IS_DOCKERIZE:
            print("{}/model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(args.io_home, problem_var, i, model_num))
            dqn_agent[i].load_model("{}/model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(args.io_home, problem_var, i, model_num))
        else:
            print("model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(problem_var, i, model_num))
            dqn_agent[i].load_model("model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(problem_var, i, model_num))

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    steps = []

    actions = [0] * agent_num
    cur_state = env.reset()
    episodic_reward = 0
    start = time.time()
    for step in range(trial_len):

        for i in range(agent_num):
            actions[i] = dqn_agent[i].act(cur_state[i])

        new_state, reward, done, _ = env.step(actions)

        for i in range(agent_num):
            new_state[i] = new_state[i]
            # dqn_agent[i].remember(cur_state[i], actions[i], reward[i], new_state[i], done)
            #
            # dqn_agent[i].replay()  # internally iterates default (prediction) model
            # dqn_agent[i].target_train()  # iterates target model

            cur_state[i] = new_state[i]
            episodic_reward += reward[i]

        if done:
            break
    print("step {}".format(step))
    ep_reward_list.append(episodic_reward)
# Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Avg Reward is ==> {}".format(avg_reward))
    print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    avg_reward_list.append(avg_reward)

    if IS_DOCKERIZE:
        if args.result_comp:
            ## add time 3, state weight 0.0, model 1000, action v2
            ft_output = pd.read_csv("{}/output/ft/-PeriodicOutput.csv".format(args.io_home))
            rl_output = pd.read_csv("{}/output/test/-PeriodicOutput.csv".format(args.io_home))

            result_comp(args, ft_output, rl_output, model_num)
    else:
        if args.resultComp:
            ## add time 3, state weight 0.0, model 1000, action v2
            ft_output = pd.read_csv("output/ft/-PeriodicOutput.csv")
            rl_output = pd.read_csv("output/test/-PeriodicOutput.csv")

            result_comp(args, ft_output, rl_output, model_num)

    return avg_reward

def sappo_test(args, trial, problem_var):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    model_num = trial

    if IS_DOCKERIZE:
        salt_scenario = args.scenario_file_path
    else:
        salt_scenario = 'data/envs/salt/doan/doan_20211207_ft.scenario.json'


    if IS_DOCKERIZE:
        if 0:
            start_time = args.start_time
            trial_len = args.end_time - args.start_time
        else:
            scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
            start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
            end_time = args.end_time if args.end_time < scenario_end else scenario_end

            trial_len = end_time - start_time
    else:
        start_time = args.testStartTime
        trial_len = args.testEndTime - args.testStartTime

    if IS_DOCKERIZE:
        if args.result_comp:
            print("Start fixed time scenario for the result compare")
            ft_simulate(args)
            print("End fixed time scenario")
    else:
        if args.resultComp:
            print("Start fixed time scenario for the result compare")
            # ft_simulate(args, trial, problem_var)
            ft_simulate(args)
            print("End fixed time scenario")

    # ft_simulate(args)

    if args.action == 'kc':
        env = SALT_SAPPO_noConst(args)
    elif args.action == 'offset':
        env = SALT_SAPPO_offset(args)

    if len(args.target_TL.split(",")) == 1:
        env = SALT_SAPPO_offset_single(args)

    agent_num = env.agent_num

    # updateTargetNetwork = 1000
    sappo_agent = []
    state_space_arr = []
    ep_agent_reward_list = []
    agent_crossName = []

    total_reward = tf.Variable(0, dtype=tf.float32)
    total_reward_summary = tf.summary.scalar('train/reward', total_reward)

    for i in range(agent_num):
        target_sa = list(env.sa_obj.keys())[i]
        state_space = env.sa_obj[target_sa]['state_space']
        state_space_arr.append(state_space)
        action_space = env.sa_obj[target_sa]['action_space']
        action_min = env.sa_obj[target_sa]['action_min']
        action_max = env.sa_obj[target_sa]['action_max']
        print(f"{target_sa}, action space {action_space}, action min {action_min}, action max {action_max}")
        sappo_agent.append(PPOAgent(args=args, state_space=state_space, action_space=action_space, action_min=action_min, action_max=action_max, agentID=i))

    if IS_DOCKERIZE:
        fn = "{}/model/sappo/SAPPO-{}-trial-{}".format(args.io_home, problem_var, model_num)
    else:
        fn = "model/sappo/SAPPO-{}-trial-{}".format(problem_var, model_num)

    sess = tf.Session()
    print("fn", fn)
    saver = tf.train.Saver()
    saver.restore(sess, fn)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    steps = []

    actions = [0] * agent_num
    cur_state = env.reset()
    episodic_reward = 0
    start = time.time()

    actions = []
    logits = []
    value_t = []
    logprobability_t = []
    sa_cycle = []

    for target_sa in env.sa_obj:
        actions.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
        logits.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
        value_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
        logprobability_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
        sa_cycle = np.append(sa_cycle, env.sa_obj[target_sa]['cycle_list'][0])

    for t in range(trial_len):

        discrete_actions = []
        for i in range(agent_num):
            actions[i], value_t[i], logprobability_t[i] = sappo_agent[i].get_action([cur_state[i]], sess)
            # print("cur_state[i]", np.round(cur_state[i],2))
            # print("actions[i]", actions[i])
            actions[i], value_t[i], logprobability_t[i] = actions[i][0], value_t[i][0], logprobability_t[i][0]

            target_sa = list(env.sa_obj.keys())[i]
            discrete_action = []
            for di in range(len(actions[i])):
                # discrete_action.append(np.digitize(actions[i][di], bins=np.linspace(-1, 1, int(env.sa_obj[target_sa]['cycle_list'][di]/15))) - 1)
                if args.action=='kc':
                    discrete_action.append(0 if actions[i][di] < args.actionp else 1)
                if args.action=='offset':
                    discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2))
            discrete_actions.append(discrete_action)

        new_state, reward, done, _ = env.step(discrete_actions)

        for i in range(agent_num):
            if t % int(sa_cycle[i] * args.controlcycle) == 0:
                cur_state[i] = new_state[i]

                episodic_reward += reward[i]

        if done:
            break

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Avg Reward is ==> {}".format(avg_reward))
    print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    avg_reward_list.append(avg_reward)

    if IS_DOCKERIZE:
        if args.result_comp:
            ## add time 3, state weight 0.0, model 1000, action v2
            ft_output = pd.read_csv("{}/output/ft/-PeriodicOutput.csv".format(args.io_home))
            rl_output = pd.read_csv("{}/output/test/-PeriodicOutput.csv".format(args.io_home))
    else:
        # if args.resultComp:
            ## add time 3, state weight 0.0, model 1000, action v2
        ft_output = pd.read_csv("output/ft/-PeriodicOutput.csv")
        rl_output = pd.read_csv("output/test/-PeriodicOutput.csv")

    total_output = result_comp(args, ft_output, rl_output, model_num)

    if IS_DOCKERIZE:
        total_output.to_csv("{}/output/test/{}_{}.csv".format(args.io_home, problem_var, model_num), encoding='utf-8-sig', index=False)
    else:
        total_output.to_csv("output/test/{}_{}.csv".format(problem_var, model_num),
                            encoding='utf-8-sig', index=False)

    return avg_reward

if __name__ == "__main__":
    # ft_simulate()
    ddqn_test(20)