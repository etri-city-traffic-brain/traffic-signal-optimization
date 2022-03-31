import sys
import os
import pandas as pd
import numpy as np
import time

from xml.etree.ElementTree import parse

from policy.ddqn import DDQN
from policy.ppo import PPOAgent
from policy.ppo_rnd import PPORNDAgent

from config import TRAIN_CONFIG

from env.salt_PennStateAction import SALT_doan_multi_PSA_test, getScenarioRelatedFilePath, getScenarioRelatedBeginEndTime
from env.sappo_noConst import SALT_SAPPO_noConst, getScenarioRelatedFilePath, getScenarioRelatedBeginEndTime
from env.sappo_offset import SALT_SAPPO_offset, getScenarioRelatedFilePath, getScenarioRelatedBeginEndTime
from env.sappo_offset_single import SALT_SAPPO_offset_single, getScenarioRelatedFilePath, getScenarioRelatedBeginEndTime
from env.sappo_green_single import SALT_SAPPO_green_single, getScenarioRelatedFilePath, getScenarioRelatedBeginEndTime
from env.sappo_green_offset_single import SALT_SAPPO_green_offset_single, getScenarioRelatedFilePath, getScenarioRelatedBeginEndTime

sys.path.append(TRAIN_CONFIG['libsalt_dir'])
import libsalt

def result_comp(args, ft_output, rl_output, model_num):
    scenario_file_path, node_file_path, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args)
    tree = parse(tss_file_path)

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

    salt_scenario = scenario_file_path
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
        target_tl_obj[target]['state_space'] = len(_lane_list)
        _lane_len.append(len(_lane_list))
    # print(target_tl_obj)

    libsalt.close()

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
    ### 시나리오 환경 세팅
    salt_scenario = args.scenario_file_path
    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    ### target tl object를 가져오기 위함
    if args.action=='kc':
        print("SAPPO KEEP OR CHANGE")
        env = SALT_SAPPO_noConst(args)
    if args.action=='offset':
        if len(args.target_TL.split(",")) == 1:
            print("SAPPO OFFSET SINGLE")
            env = SALT_SAPPO_offset_single(args)
        else:
            print("SAPPO OFFSET")
            env = SALT_SAPPO_offset(args)
    if args.action=='gr':
        print("SAPPO GREEN RATIO")
        env = SALT_SAPPO_green_single(args)
    if args.action=='gro':
        print("SAPPO GREEN RATIO + OFFSET")
        env = SALT_SAPPO_green_offset_single(args)

    ### 가시화 서버용 교차로별 고정 시간 신호 기록용
    output_ft_dir = f'{args.io_home}/output/{args.mode}'
    fn_ft_phase_reward_output = f"{output_ft_dir}/ft_phase_reward_output.txt"
    f = open(fn_ft_phase_reward_output, mode='w+', buffering=-1, encoding='utf-8', errors=None, newline=None,
                              closefd=True, opener=None)
    f.write('step,tl_name,actions,phase,reward\n')
    f.close()

    ### 교차로별 고정 시간 신호 기록하면서 시뮬레이션
    libsalt.start(salt_scenario)
    libsalt.setCurrentStep(start_time)
    f = open(fn_ft_phase_reward_output, mode='a+', buffering=-1, encoding='utf-8', errors=None,
             newline=None,
             closefd=True, opener=None)
    for i in range(trial_len):
        libsalt.simulationStep()
        for target_tl in list(env.target_tl_obj.keys()):
            tlid = target_tl
            f.write("{},{},{},{},{}\n".format(libsalt.getCurrentStep(), env.target_tl_obj[target_tl]['crossName'], 0,
                                              libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid), 0))
    f.close()

    print("ft_step {}".format(libsalt.getCurrentStep()))
    libsalt.close()

def ddqn_test(args, model_num, problem_var):
    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    if args.result_comp:
        print("Start fixed time scenario for the result compare")
        ft_simulate(args)
        print("End fixed time scenario")

    problem = "SALT_doan_multi_PSA_test"
    env = SALT_doan_multi_PSA_test(args)

    agent_num = env.agent_num
    action_mask = env.action_mask

    dqn_agent = []
    state_space_arr = []
    for i in range(agent_num):
        target_tl = list(env.target_tl_obj.keys())[i]
        state_space = env.target_tl_obj[target_tl]['state_space']
        state_space_arr.append(state_space)
        action_space = env.target_tl_obj[target_tl]['action_space']
        dqn_agent.append(DDQN(args=args, env=env, state_space=state_space, action_space=action_space, epsilon=0, epsilon_min=0))
        print("{}/model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(args.io_home, problem_var, i, model_num))
        dqn_agent[i].load_model("{}/model/ddqn/PSA-{}-agent{}-trial-{}.h5".format(args.io_home, problem_var, i, model_num))

    # To store reward history of each episode
    ep_reward_list = []

    actions = [0] * agent_num
    cur_state = env.reset()
    episodic_reward = 0
    start = time.time()
    for step in range(trial_len):
        for i in range(agent_num):
            actions[i] = dqn_agent[i].act(cur_state[i])

        new_state, reward, done, _ = env.step(actions)

        for i in range(agent_num):
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

    if args.result_comp:
        ft_output = pd.read_csv("{}/output/simulate/-PeriodicOutput.csv".format(args.io_home))
        rl_output = pd.read_csv("{}/output/test/-PeriodicOutput.csv".format(args.io_home))

        result_comp(args, ft_output, rl_output, model_num)

    return avg_reward

def sappo_test(args, model_num, problem_var):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    if args.result_comp:
        print("Start fixed time scenario for the result compare")
        ft_simulate(args)
        print("End fixed time scenario")

    if args.action=='kc':
        print("SAPPO KEEP OR CHANGE")
        env = SALT_SAPPO_noConst(args)
    if args.action=='offset':
        if len(args.target_TL.split(",")) == 1:
            print("SAPPO OFFSET SINGLE")
            env = SALT_SAPPO_offset_single(args)
        else:
            print("SAPPO OFFSET")
            env = SALT_SAPPO_offset(args)
    if args.action=='gr':
        print("SAPPO GREEN RATIO")
        env = SALT_SAPPO_green_single(args)
    if args.action=='gro':
        print("SAPPO GREEN RATIO + OFFSET")
        env = SALT_SAPPO_green_offset_single(args)

    agent_num = env.agent_num

    ppo_agent = []
    state_space_arr = []

    for i in range(agent_num):
        target_sa = list(env.sa_obj.keys())[i]
        state_space = env.sa_obj[target_sa]['state_space']
        state_space_arr.append(state_space)
        action_space = env.sa_obj[target_sa]['action_space']
        action_min = env.sa_obj[target_sa]['action_min']
        action_max = env.sa_obj[target_sa]['action_max']
        print(f"{target_sa}, action space {action_space}, action min {action_min}, action max {action_max}")
        ppo_agent.append(PPOAgent(args=args, state_space=state_space, action_space=action_space, action_min=action_min, action_max=action_max, agentID=i))

    fn = "{}/model/sappo/SAPPO-{}-trial-{}".format(args.io_home, problem_var, model_num)

    sess = tf.Session()
    print("fn", fn)
    saver = tf.train.Saver()
    saver.restore(sess, fn)

    # To store reward history of each episode
    ep_reward_list = []

    cur_state = env.reset()
    episodic_reward = 0
    start = time.time()

    actions, v_t, logp_t = [], [], []

    sa_cycle = []
    for target_sa in env.sa_obj:
        actions.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
        v_t.append([0])
        logp_t.append([0])

        sa_cycle = np.append(sa_cycle, env.sa_obj[target_sa]['cycle_list'][0])

    for t in range(trial_len):
        discrete_actions = []
        for i in range(agent_num):
            actions[i], v_t[i], logp_t[i] = ppo_agent[i].get_action([cur_state[i]], sess)
            actions[i], v_t[i], logp_t[i] = actions[i][0], v_t[i][0], logp_t[i][0]

            target_sa = list(env.sa_obj.keys())[i]
            discrete_action = []

            for di in range(len(actions[i])):
                if args.action == 'kc':
                    discrete_action.append(0 if actions[i][di] < args.actionp else 1)
                if args.action == 'offset':
                    # discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2))
                    discrete_action.append(int(np.round(actions[i][di] * sa_cycle[i]) / 2 / args.offsetrange))
                if args.action == 'gr':
                    discrete_action.append(np.digitize(actions[i][di], bins=np.linspace(-1, 1, len(env.sa_obj[target_sa]['action_list_list'][di]))) - 1)

            if args.action == 'gro':
                for di in range(int(len(actions[i]) / 2)):
                    discrete_action.append(int(np.round(actions[i][di * 2] * sa_cycle[i]) / 2 / args.offsetrange))
                    discrete_action.append(np.digitize(actions[i][di * 2 + 1], bins=np.linspace(-1, 1, len(env.sa_obj[target_sa]['action_list_list'][di]))) - 1)

            discrete_actions.append(discrete_action)

        new_state, reward, done, _ = env.step(discrete_actions)

        for i in range(agent_num):
            # if t % int(sa_cycle[i] * args.controlcycle) == 0:
            cur_state[i] = new_state[i]
            episodic_reward += reward[i]

        if done:
            break

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Avg Reward is ==> {}".format(avg_reward))
    print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    if args.result_comp:
        ## add time 3, state weight 0.0, model 1000, action v2
        ft_output = pd.read_csv("{}/output/simulate/-PeriodicOutput.csv".format(args.io_home))
        rl_output = pd.read_csv("{}/output/test/-PeriodicOutput.csv".format(args.io_home))

    total_output = result_comp(args, ft_output, rl_output, model_num)
    total_output.to_csv("{}/output/test/{}_{}.csv".format(args.io_home, problem_var, model_num), encoding='utf-8-sig', index=False)

    return avg_reward

def ppornd_test(args, model_num, problem_var):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    scenario_begin, scenario_end = getScenarioRelatedBeginEndTime(args.scenario_file_path)
    start_time = args.start_time if args.start_time > scenario_begin else scenario_begin
    end_time = args.end_time if args.end_time < scenario_end else scenario_end
    trial_len = end_time - start_time

    if args.result_comp:
        print("Start fixed time scenario for the result compare")
        ft_simulate(args)
        print("End fixed time scenario")

    if args.action == 'kc':
        env = SALT_SAPPO_noConst(args)
    elif args.action == 'offset':
        env = SALT_SAPPO_offset(args)

    if len(args.target_TL.split(",")) == 1:
        env = SALT_SAPPO_offset_single(args)

    agent_num = env.agent_num

    ppornd_agent = []
    state_space_arr = []

    for i in range(agent_num):
        target_sa = list(env.sa_obj.keys())[i]
        state_space = env.sa_obj[target_sa]['state_space']
        state_space_arr.append(state_space)
        action_space = env.sa_obj[target_sa]['action_space']
        action_min = env.sa_obj[target_sa]['action_min']
        action_max = env.sa_obj[target_sa]['action_max']
        print(f"{target_sa}, action space {action_space}, action min {action_min}, action max {action_max}")
        ppornd_agent.append(PPORNDAgent(args=args, state_space=state_space, action_space=action_space, action_min=action_min, action_max=action_max, agentID=i))

    fn = "{}/model/ppornd/PPORND-{}-trial-{}".format(args.io_home, problem_var, model_num)

    sess = tf.Session()
    print("fn", fn)
    saver = tf.train.Saver()
    saver.restore(sess, fn)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes

    cur_state = env.reset()
    episodic_reward = 0
    start = time.time()

    actions = []
    value_t = []
    logprobability_t = []
    sa_cycle = []

    for target_sa in env.sa_obj:
        actions.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
        value_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
        logprobability_t.append([0] * env.sa_obj[target_sa]['action_space'].shape[0])
        sa_cycle = np.append(sa_cycle, env.sa_obj[target_sa]['cycle_list'][0])

    for t in range(trial_len):

        discrete_actions = []
        for i in range(agent_num):
            actions[i] = ppornd_agent[i].choose_action([cur_state[i]], sess)
            target_sa = list(env.sa_obj.keys())[i]
            discrete_action = []
            for di in range(len(actions[i])):
                # discrete_action.append(np.digitize(actions[i][di], bins=np.linspace(-1, 1, int(env.sa_obj[target_sa]['cycle_list'][di]/15))) - 1)
                if args.action=='kc':
                    discrete_action.append(0 if actions[i][di] < args.actionp else 1)
                if args.action=='offset':
                    # discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2))
                    discrete_action.append(int(np.round(actions[i][di]*sa_cycle[i])/2/args.offsetrange))
            discrete_actions.append(discrete_action)

        new_state, reward, done, _ = env.step(discrete_actions)

        for i in range(agent_num):
            cur_state[i] = new_state[i]
            episodic_reward += reward[i]

        if done:
            break

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Avg Reward is ==> {}".format(avg_reward))
    print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    if args.result_comp:
        ## add time 3, state weight 0.0, model 1000, action v2
        ft_output = pd.read_csv("{}/output/simulate/-PeriodicOutput.csv".format(args.io_home))
        rl_output = pd.read_csv("{}/output/test/-PeriodicOutput.csv".format(args.io_home))

    total_output = result_comp(args, ft_output, rl_output, model_num)
    total_output.to_csv("{}/output/test/{}_{}.csv".format(args.io_home, problem_var, model_num), encoding='utf-8-sig', index=False)

    return avg_reward

if __name__ == "__main__":
    # ft_simulate()
    ddqn_test(20)