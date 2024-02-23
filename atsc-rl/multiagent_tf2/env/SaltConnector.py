
import json
import numpy as np
import os
import pandas as pd
import platform
import pprint

from gym import spaces
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse


import libsalt
from env.TrafficEnvironmentConnector import TrafficEnvironmentConnector
from DebugConfiguration import DBG_OPTIONS
from TSOConstants import _RESULT_COMPARE_SKIP_
from TSOUtil import getActionList
from TSOUtil import getPossibleActionList


class SaltConnector(TrafficEnvironmentConnector):
    def __init__(self, name="salt"):
        super(SaltConnector, self).__init__(type(self).__name__)

    #####
    ##
    ## control simulator
    ##
    def start(self, param1, output_home="."):
        libsalt.start(param1, output_home)


    def close(self):
        return libsalt.close()


    def increaseStep(self):
        return libsalt.simulationStep()



    #####
    ##
    ## set
    ##

    def setCurrentStep(self, step):
        libsalt.setCurrentStep(step)


    # def setPhase(self, tl, given_phase):
    #     '''
    #     set the signal phase of the given traffic light as a given phase
    #     ''''
    #     step = libsalt.getCurrentStep()
    #     current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tl)
    #     libsalt.trafficsignal.changeTLSPhase(step, tl, current_schedule_id, np.int(given_phase))



    def setPhase(self, sim_step, tlid, schedule_id, tl_phase):
        ''' 
        set the signal phase of the given traffic light as a given phase
        '''
        libsalt.trafficsignal.changeTLSPhase(sim_step, tlid, schedule_id, np.int(tl_phase))



    def setPhaseVector(self, sim_step, tlid, scheduleId, phase_vector):
        libsalt.trafficsignal.setTLSPhaseVector(sim_step, tlid, scheduleId, phase_vector)


    #####
    ##
    ##  get
    ##

    def getAverageDensityOfLane(self, lane_id):
        return libsalt.lane.getAverageDensity(lane_id)


    def getAverageSpeedOfLink(self, link_id):
        return libsalt.link.getAverageSpeed(link_id)


    def getCurrentPhaseIndex(self, tlid):
        return libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)


    def getCurrentScheduleId(self, tlid):
        return libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)


    def getCurrentSchedulePhaseVector(self, tlid):
        #  [(26, 'rrrrrgrrGgrrrrrrGGG'), (4, 'rrrrryrryyrrrrrryyg'), (72, 'GGGGrrrrrrGGGGrrrrG'), (3, 'yyyyrrrrrryyyyrrrry'),
        #        (17, 'rrrrGgrrrgrrrrGrrrr'), (3, 'rrrryyrrryrrrryrrrr'), (51, 'rrrrrrGGrrrrrrrGrrr'), (4, 'rrrrrryyrrrrrrryrrr')]
        return libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector


    def getCurrentStep(self):
        return libsalt.getCurrentStep()

    def getLengthOfLane(self, lain_id):
        return libsalt.lane.getLength(lane_id)

    def getNumLaneOfLink(self, edge):
        return libsalt.link.getNumLane(edge)

    def getNumVehPassedOfLane(self, lane_id):
        return libsalt.lane.getNumVehPassed(lane_id)


    ## ing.... ing....
    ## 인자로 (sa_obj, sa_id)가 오면 sa 에 대한 정보를
    ## 인자로 (tl_obj, tl_id)가 오면 tl 에 대한 정보를 반환한다.
    ##
    ## RewardMgmt::__getRewardInfoPerTL() 와 함께....
    def getSumPassed(self, gathered_info, an_info_dic):
        link_list_0 = an_info_dic['in_edge_list_0']
        lane_list_0 = an_info_dic['in_lane_list_0']
        for l in link_list_0:
            gathered_info = np.append(gathered_info, libsalt.link.getSumPassed(l))

        return gathered_info


    def getSumPassedOfLink(self, link_id):
        return libsalt.link.getSumPassed(link_id)


    def getSumTravelTimeOfLink(self, link_id):
        return libsalt.link.getSumTravelTime(link_id)



    def getTravelTime(self, gathered_info, an_info_dic, reward_info_collection_cycle):
        link_list_0 = an_info_dic['in_edge_list_0']
        lane_list_0 = an_info_dic['in_lane_list_0']
        for l in link_list_0:
            gathered_info = np.append(gathered_info, libsalt.link.getSumTravelTime(l) /
                                      (len(link_list_0) * reward_info_collection_cycle))
        return gathered_info


    def getWaitingQLength(self, gathered_info, an_info_dic):
        link_list_0 = an_info_dic['in_edge_list_0']
        lane_list_0 = an_info_dic['in_lane_list_0']
        for l in link_list_0:
            gathered_info = np.append(gathered_info,
                                    libsalt.link.getAverageWaitingQLength(l) * sum([l in x for x in lane_list_0]))
        return gathered_info


    def getWaitingTime(self, gathered_info, an_info_dic, action_t):
        link_list_0 = an_info_dic['in_edge_list_0']
        lane_list_0 = an_info_dic['in_lane_list_0']
        for l in link_list_0:
            gathered_info = np.append(gathered_info, libsalt.link.getAverageWaitingTime(l) / action_t)
        return gathered_info





    #####
    ##
    ## methods to handle SALT related info
    ##
    ##
    ## from SaltEnvUtil.py
    def getSimulationStartStepAndEndStep(self, args):
        '''
        get begin- & end-time of simulation

        :param args:
        :return:
        '''
        abs_scenario_file_path = '{}/{}'.format(os.getcwd(), args.scenario_file_path)

        with open(abs_scenario_file_path, 'r') as json_file:
            json_data = json.load(json_file)
            scenario_begin = json_data["scenario"]["time"]["begin"]
            scenario_end = json_data["scenario"]["time"]["end"]

        start_step = args.start_time if args.start_time > scenario_begin else scenario_begin
        end_step = args.end_time if args.end_time < scenario_end else scenario_end
        return start_step, end_step

    ##


    def __processStatisticalInformation(self, field, op, op2, ft_0, ft_all, rl_0, rl_all, individual_output):
        '''
        process statistics info to calculate improvement rate

        :param field: filed name of interesting statistics info
        :param op: 1 or -1
        :param op2: "sum" or "mean"
        :param ft_0: DataFrame object which contains statistics information (0-hop, fixed signal control)
        :param ft_all: DataFrame object which contains statistics information (0-hop & 1-hop, fixed signal control)
        :param rl_0: DataFrame object which contains statistics information (0-hop, inference-based signal control)
        :param rl_all: DataFrame object which contains statistics information (0-hop & 1-hop, inference-based signal control)
        :param individual_output: processed output
        :return:
        '''
        op_dic = {"sum": np.sum, "mean": np.mean}

        ft_passed = op_dic[op2](ft_0[field])  # np.sum(ft_0[field]) or np.mean(ft_0[field])
        rl_passed = op_dic[op2](rl_0[field])
        # try :
        #     import warnings
        #     warnings.filterwarnings('error')
        #     imp = op * (rl_passed - ft_passed) / ft_passed * 100
        # except Warning:
        #     print("30  rl_passed={}, ft_passed={}".format(rl_passed, ft_passed))

        # todo should care when ft_passed is 0
        #               how about adding very small value(0.00000001)
        #               ft_passed += 0.00000001
        if ft_passed == 0.0:
            imp = 0.0
        else:
            imp = op * (rl_passed - ft_passed) / ft_passed * 100
        ft_passed = np.round(ft_passed, 2)
        rl_passed = np.round(rl_passed, 2)
        imp = np.round(imp, 2)

        if DBG_OPTIONS.PrintResultCompare:
            print("0-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(field, op2, ft_passed,
                                                                              field, op2, rl_passed, imp))
        individual_output = pd.concat(
            [individual_output, pd.DataFrame({'ft_{}_{}_0hop'.format(field, op2): [ft_passed],
                                              'rl_{}_{}_0hop'.format(field, op2): [rl_passed],
                                              'imp_{}_{}_0hop'.format(field, op2): [imp]})], axis=1)

        ft_passed = op_dic[op2](ft_all[field])
        rl_passed = op_dic[op2](rl_all[field])
        ft_passed = np.round(ft_passed, 2)
        rl_passed = np.round(rl_passed, 2)
        # try :
        #     import warnings
        #     warnings.filterwarnings('error')
        #     imp = op * (rl_passed - ft_passed) / ft_passed * 100
        # except Warning:
        #     print("52  rl_passed={}, ft_passed={}".format(rl_passed, ft_passed))

        # todo should care when ft_passed is 0
        #               how about adding very small value(0.00000001)
        #               ft_passed += 0.00000001
        if ft_passed == 0.0:
            imp = 0.0
        else:
            imp = op * (rl_passed - ft_passed) / ft_passed * 100
        imp = np.round(imp, 2)

        if DBG_OPTIONS.PrintResultCompare:
            print("1-hop lanes Fixed Time {} {} {} RL {} {} {} Imp {}".format(field, op2, ft_passed,
                                                                              field, op2, rl_passed, imp))
        individual_output = pd.concat(
            [individual_output, pd.DataFrame({'ft_{}_{}_1hop'.format(field, op2, ): [ft_passed],
                                              'rl_{}_{}_1hop'.format(field, op2, ): [rl_passed],
                                              'imp_{}_{}_1hop'.format(field, op2, ): [imp]})], axis=1)
        return individual_output


    def __getStatisticsInformationAboutGivenEdgeList(self, ft_output, rl_output, in_edge_list_0, in_edge_list, cut_interval):
        '''
        get statistics information which are related to given edge list

        :param ft_output: DataFrame object which contains statistics information about traffic simulation using fixed signals to control traffic lights
        :param rl_output:  DataFrame object which contains statistics information about traffic simulation using inference to control traffic lights
        :param in_edge_list_0: edge list with 0-hop
        :param in_edge_list: edge list with 0-, 1-hop
        :param cut_interval: last time to delete statistics info
        :return:
        '''
        ft_output2 = ft_output[ft_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list_0) + '$', na=False)]
        rl_output2 = rl_output[rl_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list_0) + '$', na=False)]
        ft_output3 = ft_output[ft_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list) + '$', na=False)]
        rl_output3 = rl_output[rl_output['roadID'].str.contains('^' + '$|^'.join(in_edge_list) + '$', na=False)]
        ft_output2 = ft_output2[ft_output2['intervalbegin'] >= cut_interval]  # 3600 초 이후의 것들만 성능 향상 계산 대상으로 한다.
        rl_output2 = rl_output2[rl_output2['intervalbegin'] >= cut_interval]
        ft_output3 = ft_output3[ft_output3['intervalbegin'] >= cut_interval]
        rl_output3 = rl_output3[rl_output3['intervalbegin'] >= cut_interval]
        return ft_output2, ft_output3, rl_output2, rl_output3


    def __compareResultInternal(self, individual_output, comp_tl_list, target_tl_obj, ft_output, rl_output, cut_interval):
        ##-- set the info to be extracted : kind, method
        ##---- kinds of information to be extracted
        if DBG_OPTIONS.IngCompResult:
            varList = ['VehPassed', 'AverageSpeed', 'WaitingTime', 'AverageDensity', 'SumTravelTime', 'AvgTravelTime',
                       'WaitingQLength']
        else:
            varList = ['VehPassed', 'AverageSpeed', 'WaitingTime', 'AverageDensity', 'SumTravelTime', 'WaitingQLength']

        ##----methods how to calculate
        ##     larger is good if this value is positive, smaller is good if this value is negative
        if DBG_OPTIONS.IngCompResult:
            varOp = [1, 1, -1, -1, -1, -1, -1]
            varOp2 = ['sum', 'mean', 'sum', 'mean', 'sum', 'mean', 'mean']
        else:
            varOp = [1, 1, -1, -1, -1, -1]
            varOp2 = ['sum', 'mean', 'sum', 'mean', 'sum', 'mean']

        in_edge_list = []
        in_edge_list_0 = []

        for tl in comp_tl_list:
            in_edge_list = np.append(in_edge_list, target_tl_obj[tl]['in_edge_list'])
            in_edge_list_0 = np.append(in_edge_list_0, target_tl_obj[tl]['in_edge_list_0'])
            # if DBG_OPTIONS.PrintResultCompare:
            #     print(target_tl_obj[tl]['crossName'], target_tl_obj[tl]['in_edge_list_0'])

        if DBG_OPTIONS.PrintResultCompare:
            # print("\nAll Target TL summary.....")
            print(f"\n{individual_output['name'][0]} summary.....")

        ft_output2, ft_output3, rl_output2, rl_output3 = \
            self.__getStatisticsInformationAboutGivenEdgeList(ft_output, rl_output, in_edge_list_0, in_edge_list, cut_interval)

        # process by information type(kind) and add it to DataFrame object
        for v in range(len(varList)):
            individual_output = self.__processStatisticalInformation(varList[v], varOp[v], varOp2[v],
                                                              ft_output2, ft_output3, rl_output2, rl_output3,
                                                              individual_output)
        return individual_output

    def __compareResult(self, args, target_tl_obj, ft_output, rl_output, model_num, passed_res_comp_skip=-1):
        '''
        compare two result files and calculate improvement rate for each intersection, each SA and overall

        :param args:
        :param target_tl_obj: information about target TL
        :param ft_output: a data frame object which was generated by reading an output (csv) file of simulator
                               that performed the signal control simulation based on the fixed signal
        :param rl_output: a data frame object which was generated by reading an output (csv) file of simulator
                               that performed signal control simulation based on reinforcement learning inference
        :param model_num: number which indicate optimal model which was used to TEST
        :param passed_res_comp_skip : steps to skip to exclude comparison(result comparison)
        :return:
        '''
        ##
        ## Various statistical information related to intersections is extracted from the DataFrame object
        ##      containing the contents of the CSV file created by the simulator.
        ##-- create empty DataFrame object
        total_output = pd.DataFrame()

        if passed_res_comp_skip == -1:
            cut_interval = args.start_time + _RESULT_COMPARE_SKIP_  # 2시간 테스트시 앞에  일정 시간은 비교대상에서 제외
        else:
            cut_interval = args.start_time + passed_res_comp_skip

        if DBG_OPTIONS.PrintResultCompare:
            print(f"training step: {args.start_time} to {args.end_time}")
            print(f"comparing step: {cut_interval} to {args.end_time}")
            print(f"model number: {model_num}")

        #
        # for each intersection
        #
        target_sa_tl_dic = {}  # to save TL info per SA
        for tl in target_tl_obj:
            if "SA " not in target_tl_obj[tl]['signalGroup']:
                target_tl_obj[tl]['signalGroup'] = 'SA ' + target_tl_obj[tl]['signalGroup']
                # add columns : crossName, signalGroup
            individual_output = pd.DataFrame(
                {'name': [target_tl_obj[tl]['crossName']], 'SA': [target_tl_obj[tl]['signalGroup']]})

            individual_output = self.__compareResultInternal(individual_output, [tl], target_tl_obj, ft_output, rl_output,
                                                        cut_interval)
            total_output = pd.concat([total_output, individual_output])

            ## gather SA info
            sa_name = target_tl_obj[tl]['signalGroup']
            if sa_name in target_sa_tl_dic.keys():
                target_sa_tl_dic[sa_name].append(tl)
            else:
                target_sa_tl_dic[sa_name] = [tl]

            if DBG_OPTIONS.PrintResultCompare:
                print(f"sa_name={sa_name}  tl_name={target_tl_obj[tl]['crossName']}  tl_node_id={tl}")

        #
        # for each SA
        #
        for sa in target_sa_tl_dic.keys():
            if DBG_OPTIONS.PrintResultCompare:
                print(f'{sa}')
                for tl in list(target_sa_tl_dic[sa]):
                    print(target_tl_obj[tl]['crossName'])

            individual_output = pd.DataFrame({'name': [sa], 'SA': ['total']})
            individual_output = self.__compareResultInternal(individual_output, list(target_sa_tl_dic[sa]), target_tl_obj,
                                                        ft_output, rl_output, cut_interval)
            total_output = pd.concat([total_output, individual_output])

        #
        # for entire target
        #
        individual_output = pd.DataFrame({'name': ['total'], 'SA': ['total']})
        individual_output = self.__compareResultInternal(individual_output, list(target_tl_obj.keys()), target_tl_obj,
                                                    ft_output, rl_output, cut_interval)
        total_output = pd.concat([total_output, individual_output])

        total_output = total_output.sort_values(by=["SA", "name"], ascending=True)

        return total_output


    def compareResult(self, args, target_tl_obj, ft_output, rl_output, model_num, passed_res_comp_skip=-1):
        '''
        compare two result files and calculate improvement rate for each intersection, each SA and overall

        :param args:
        :param target_tl_obj: information about target TL
        :param ft_output: a data frame object which was generated by reading an output (csv) file of simulator
                               that performed the signal control simulation based on the fixed signal
        :param rl_output: a data frame object which was generated by reading an output (csv) file of simulator
                               that performed signal control simulation based on reinforcement learning inference
        :param model_num: number which indicate optimal model which was used to TEST
        :param passed_res_comp_skip : steps to skip to exclude comparison(result comparison)
        :return:
        '''
        if DBG_OPTIONS.IngCompResult:

            # todo ft_output과 rl_output에  AvgTravelTime 추가
            #   AvgTravelTime = SumTravelTime / VehPassed
            #     ['VehPassed', 'AverageSpeed', 'WaitingTime', 'AverageDensity', 'SumTravelTime', 'AvgTravelTime', 'WaitingQLength']

            #     frame['avg']=frame['kor']/frame['n']
            # todo NaN을 0으로 바꾸기, inf를 0으로 바꾸기
            ft_output['AvgTravelTime'] = ft_output['SumTravelTime'] / ft_output['VehPassed']
            for i in ft_output[ft_output['VehPassed'] == 0.0].index:
                ft_output.at[i, 'AvgTravelTime'] = 0.0

            rl_output['AvgTravelTime'] = rl_output['SumTravelTime'] / rl_output['VehPassed']
            for i in rl_output[rl_output['VehPassed'] == 0.0].index:
                rl_output.at[i, 'AvgTravelTime'] = 0.0

        return self.__compareResult(args, target_tl_obj, ft_output, rl_output, model_num, passed_res_comp_skip)





    def __getScenarioRelatedFilePath(self, scenario_file_path):
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



    def __getScheduleID(self, traffic_signal, given_start_time):
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




    def __constructTSSRelatedInfo(self, args, tss_file_path, sa_name_list):
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
                nid = x.attrib['nodeID']
                target_tl_obj[nid] = {}
                target_tl_obj[nid]['crossName'] = x.attrib['crossName']

                # to make same format : "101", "SA101", "SA 101" ==> "SA 101"
                _signalGroup = x.attrib['signalGroup'].strip()
                if 0:
                    if "SA" not in _signalGroup:
                        _signalGroup = "SA " + _signalGroup
                    if "SA " not in _signalGroup:
                        _signalGroup = _signalGroup.replace("SA", "SA ")
                else:
                    _signalGroup = _signalGroup.replace("SA", "")  # remove "SA"
                    _signalGroup.strip()        # remove leading & tailing whitespace
                    _signalGroup = "SA " + _signalGroup     # add "SA "

                target_tl_obj[nid]['signalGroup'] = _signalGroup

                s_id = self.__getScheduleID(x, args.start_time)

                # print(_signalGroup)
                target_tl_obj[nid]['offset'] = int(x.find(f"schedule[@id='{s_id}']").attrib['offset'])
                target_tl_obj[nid]['minDur'] = \
                    [int(y.attrib['minDur']) if 'minDur' in y.attrib else int(y.attrib['duration']) for
                                                                y in x.findall(f"schedule[@id='{s_id}']/phase")]
                target_tl_obj[nid]['maxDur'] = \
                    [int(y.attrib['maxDur']) if 'maxDur' in y.attrib else int(y.attrib['duration']) for
                                                                y in x.findall(f"schedule[@id='{s_id}']/phase")]
                target_tl_obj[nid]['cycle'] = \
                    np.sum([int(y.attrib['duration']) for y in x.findall(f"schedule[@id='{s_id}']/phase")])
                target_tl_obj[nid]['duration'] = \
                    [int(y.attrib['duration']) for y in x.findall(f"schedule[@id='{s_id}']/phase")]
                tmp_duration_list = np.array([int(y.attrib['duration']) for y in x.findall(f"schedule[@id='{s_id}']/phase")])

                if 0:
                    target_tl_obj[nid]['green_idx'] = np.where(tmp_duration_list > 5)
                else:
                    # -- todo bug.... minDur과 maxDur이 같은 경우에  green_idx로 분류되지 않을 가능성이 있다.
                    # ---- sg = 37 nodeid = cluster_554800075_554800077_554800078_554800080_554801615_554801640_554805918_554805919_554813394
                    # ----     minDur과 maxDur이 동일한 Phase가  지속시간이 가장 긴 max_phase인 경우로 max_phase를 찾지못해서 IndexErr Exception이 발생한다.
                    target_tl_obj[nid]['green_idx'] = \
                        np.where(np.array(target_tl_obj[nid]['minDur']) != np.array(target_tl_obj[nid]['maxDur']))

                ### for select discrete action with the current phase ratio from tanH Prob.
                dur_arr = []
                for g in target_tl_obj[nid]['green_idx'][0]:
                    dur_arr.append(target_tl_obj[nid]['duration'][g])
                dur_ratio = dur_arr / np.sum(dur_arr)
                tmp = -1
                dur_bins = []
                for dr in dur_ratio:
                    dur_bins.append(tmp + dr * 2)
                    tmp += dr * 2
                # print(target_tl_obj[nid]['green_idx'])
                target_tl_obj[nid]['duration_bins'] = dur_bins
                target_tl_obj[nid]['main_green_idx'] = \
                    np.where(tmp_duration_list == np.max(tmp_duration_list))
                target_tl_obj[nid]['sub_green_idx'] = \
                    list(set(target_tl_obj[nid]['green_idx'][0]) -
                         set(np.where(tmp_duration_list == np.max(tmp_duration_list))[0]))
                target_tl_obj[nid]['tl_idx'] = i
                target_tl_obj[nid]['remain'] = \
                    target_tl_obj[nid]['cycle'] - np.sum(target_tl_obj[nid]['minDur'])
                target_tl_obj[nid]['max_phase'] = \
                    np.where(target_tl_obj[nid]['green_idx'][0] ==target_tl_obj[nid]['main_green_idx'][0][0])
                target_tl_obj[nid]['action_space'] = len(target_tl_obj[nid]['green_idx'][0])
                target_tl_obj[nid]['action_list'] = \
                    getPossibleActionList(args, target_tl_obj[nid]['duration'],
                                            target_tl_obj[nid]['minDur'],
                                            target_tl_obj[nid]['maxDur'],
                                            target_tl_obj[nid]['green_idx'],
                                            getActionList(len(target_tl_obj[nid]['green_idx'][0]),
                                            target_tl_obj[nid]['max_phase'][0][0]))
                i += 1

        return target_tl_obj


    def __constructEdgeRelatedInfo(self, edge_file_path, target_tl_id_list, target_tl_obj):
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



    def __constructLaneRelatedInfo(self, args, salt_scenario, target_tl_obj):
        '''
        construct LANE related info from given traffic environment
        :param atgs : parsed argument
        :param salt_scenario: scenario file path
        :param target_tl_obj: an object to store constructed LANE related info
        :return:
        '''
        startStep = 0

        self.start(salt_scenario, args.output_home)
        self.setCurrentStep(startStep)

        _lane_len = []
        for target in target_tl_obj:
            _lane_list = []
            _lane_list_0 = []
            for edge in target_tl_obj[target]['in_edge_list_0']:
                for lane in range(self.getNumLaneOfLink(edge)):
                    _lane_id = "{}_{}".format(edge, lane)
                    _lane_list.append(_lane_id)
                    _lane_list_0.append((_lane_id))
                    # print(_lane_id, te_conn.getLengthOfLane(_lane_id))
            target_tl_obj[target]['in_lane_list_0'] = _lane_list_0
            _lane_list_1 = []
            for edge in target_tl_obj[target]['in_edge_list_1']:
                for lane in range(self.getNumLaneOfLink(edge)):
                    _lane_id = "{}_{}".format(edge, lane)
                    _lane_list.append(_lane_id)
                    _lane_list_1.append((_lane_id))
                    # print(_lane_id, te_conn.getLengthOfLane(_lane_id))
            target_tl_obj[target]['in_lane_list_1'] = _lane_list_1
            target_tl_obj[target]['in_lane_list'] = _lane_list
            if args.state == 'vd':
                target_tl_obj[target]['state_space'] = len(_lane_list_0) * 2 + 1
            else:
                target_tl_obj[target]['state_space'] = len(_lane_list_0) + 1
            _lane_len.append(len(_lane_list))

        self.close()

        return target_tl_obj, _lane_len



    ### 신호 최적화 대상 교차로 및 교차로 그룹에 대한 정보를 object로 생성
    def getSaRelatedInfo(self, args, sa_name_list, salt_scenario):
        '''
        gather SA related info such as contained TLs, TSS, lane, link,....
        :param args: parsed argument
        :param sa_name_list: list of name of SA which are interesting
        :param salt_scenario: scenario file path
        :return:
        '''

        _, _, edge_file_path, tss_file_path = self.__getScenarioRelatedFilePath(args.scenario_file_path)

        target_tl_obj = self.__constructTSSRelatedInfo(args, tss_file_path, sa_name_list)

        ## get the identifier of target intersection to optimize signal
        target_tl_id_list = list(target_tl_obj.keys())

        ## get EDGE info which are belong to the target intersection group for optimizing signal
        target_tl_obj = self.__constructEdgeRelatedInfo(edge_file_path, target_tl_id_list, target_tl_obj)

        ## build incomming LANE related info by executing the simulator
        target_tl_obj, _lane_len = self.__constructLaneRelatedInfo(args, salt_scenario, target_tl_obj)

        if DBG_OPTIONS.PrintSaRelatedInfo:
            print("target_tl_obj")
            pprint.pprint(target_tl_obj, width=200, compact=True)

        ### for SAPPO Agent ###
        sa_obj = {}
        for tl_obj in target_tl_obj:
            if target_tl_obj[tl_obj]['signalGroup'] not in sa_obj:
                sa_obj[target_tl_obj[tl_obj]['signalGroup']] = {}
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['crossName_list'] = []  # 교차로 그룹에 속한 교차로 이름 목록
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tlid_list'] = []  # 교차로 id 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space'] = 0  # state space - 교차로 그룹에 속한 교차로의 in_lane 수

                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'] = []  # action_min에 대한 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'] = []  # action_max에 대한 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['offset_list'] = []  # 각 교차로의 offset 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['minDur_list'] = []  # 각 교차로의 최소 녹색 시간 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['maxDur_list'] = []  # 각 교차로의 최대 녹색 시간 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['cycle_list'] = []  # 각 교차로의 주기 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_list'] = []  # 각 교차로의 현재 신호 시간 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['green_idx_list'] = []  # 각 교차로의 녹색 시간 인덱스 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_bins_list'] = []  # 각 교차로의 현재 신호 비율에 따라 -1에서 1까지 등분한 리스트(ppo phase 선택 action용)
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['main_green_idx_list'] = []  # 각 교차로의 주 현시 인덱스 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['sub_green_idx_list'] = []  # 각 교차로의 나머지 현시 인덱스 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tl_idx_list'] = []  # 각 교차로의 tl_idx 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['remain_list'] = []  # 각 교차로의 잔여 녹색 시간 리스트(잔여 녹색 시간 = 주기 - 최소 녹색 시간의 합)
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['max_phase_list'] = []  # 각 교차로의 녹색 현시가 가장 긴 현시 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space_list'] = []  # 각 교차로의 action space 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_list_list'] = []  # 각 교차로의 녹색 시간 조정 action list(주 현시와 나머지 현시 조정)
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space_list'] = []  # 각 교차로의 state space 리스트
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list'] = []  # 각 교차로의 진입 link list(0-hop, 1-hop), 2차원으로 구분 없이 1차원으로 모든 link
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0'] = []  # 각 교차로의 진입 link list(0-hop), 2차원으로 구분 없이 1차원으로 모든 link
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1'] = []  # 각 교차로의 진입 link list(1-hop), 2차원으로 구분 없이 1차원으로 모든 link
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_list'] = []  # 각 교차로의 진입 link list(0-hop, 1-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0_list'] = []  # 각 교차로의 진입 link list(0-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1_list'] = []  # 각 교차로의 진입 link list(1-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list'] = []  # 각 교차로의 진입 lane list(0-hop, 1-hop), 2차원으로 구분 없이 1차원으로 모든 link
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0'] = []  # 각 교차로의 진입 lane list(0-hop), 2차원으로 구분 없이 1차원으로 모든 link
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1'] = []  # 각 교차로의 진입 lane list(1-hop), 2차원으로 구분 없이 1차원으로 모든 link
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_list'] = []  # 각 교차로의 진입 lane list(0-hop, 1-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0_list'] = []  # 각 교차로의 진입 lane list(0-hop), 2차원으로 구분하여 각 교차로마다 구분 됨
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1_list'] = []  # 각 교차로의 진입 lane list(1-hop), 2차원으로 구분하여 각 교차로마다 구분 됨

            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['crossName_list'].append(target_tl_obj[tl_obj]['crossName'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tlid_list'].append(tl_obj)
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space'] += target_tl_obj[tl_obj]['state_space']
            if args.action == 'gro':
                # todo should check correctness of value : 0..1,   .. (# of green phase  -1)
                # for offset
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'].append(0)
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'].append(
                    target_tl_obj[tl_obj]['action_space'] - 1)

                # for green ratio
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'].append(0)
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'].append(
                    target_tl_obj[tl_obj]['action_space'] - 1)

            else:
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min'].append(0)
                sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max'].append(
                    target_tl_obj[tl_obj]['action_space'] - 1)

            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['offset_list'].append(target_tl_obj[tl_obj]['offset'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['minDur_list'].append(target_tl_obj[tl_obj]['minDur'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['maxDur_list'].append(target_tl_obj[tl_obj]['maxDur'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['cycle_list'].append(target_tl_obj[tl_obj]['cycle'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_list'].append(target_tl_obj[tl_obj]['duration'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['green_idx_list'].append(target_tl_obj[tl_obj]['green_idx'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['duration_bins_list'].append(
                target_tl_obj[tl_obj]['duration_bins'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['main_green_idx_list'].append(
                target_tl_obj[tl_obj]['main_green_idx'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['sub_green_idx_list'].append(
                target_tl_obj[tl_obj]['sub_green_idx'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tl_idx_list'].append(target_tl_obj[tl_obj]['tl_idx'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['remain_list'].append(target_tl_obj[tl_obj]['remain'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['max_phase_list'].append(target_tl_obj[tl_obj]['max_phase'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space_list'].append(
                target_tl_obj[tl_obj]['action_space'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_list_list'].append(
                target_tl_obj[tl_obj]['action_list'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['state_space_list'].append(
                target_tl_obj[tl_obj]['state_space'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list'] += target_tl_obj[tl_obj]['in_edge_list']
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0'] += target_tl_obj[tl_obj]['in_edge_list_0']
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1'] += target_tl_obj[tl_obj]['in_edge_list_1']
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_list'].append(
                target_tl_obj[tl_obj]['in_edge_list'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_0_list'].append(
                target_tl_obj[tl_obj]['in_edge_list_0'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_edge_list_1_list'].append(
                target_tl_obj[tl_obj]['in_edge_list_1'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list'] += target_tl_obj[tl_obj]['in_lane_list']
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0'] += target_tl_obj[tl_obj]['in_lane_list_0']
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1'] += target_tl_obj[tl_obj]['in_lane_list_1']
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_list'].append(
                target_tl_obj[tl_obj]['in_lane_list'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_0_list'].append(
                target_tl_obj[tl_obj]['in_lane_list_0'])
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['in_lane_list_1_list'].append(
                target_tl_obj[tl_obj]['in_lane_list_1'])

            sa_action_min = sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_min']
            sa_action_max = sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_max']
            sa_obj[target_tl_obj[tl_obj]['signalGroup']]['action_space'] = spaces.Box(low=np.array(sa_action_min),
                                                                                      high=np.array(sa_action_max),
                                                                                      dtype=np.int32)

        if DBG_OPTIONS.PrintSaRelatedInfo:
            print("sa_obj")
            pprint.pprint(sa_obj, width=200, compact=True)

        return target_tl_obj, sa_obj, _lane_len



    def gatherTsoOutputInfo(self, tl_id, tl_obj, num_hop=0):
        '''
        gather TSO-related information of given intersection

        TSO-related information : average speed, travel time, passed vehicle num

        :param tl_id: inersection identifier
        :param tl_obj:  objects which holds TL information
        :param num_hop: number of hop to calculate speed
        :return:
        '''
        link_list = tl_obj[tl_id]['in_edge_list_0']

        if num_hop > 0:
            link_list += tl_obj[tl_id]['in_edge_list_1']
        link_speed_list = []
        link_avg_time_list = []
        sum_travel_time = 0
        sum_passed = 0
        for link_id in link_list:
            # average speed
            link_speed_list.append(self.getAverageSpeedOfLink(link_id))

            # passed vehicle num
            passed = self.getSumPassedOfLink(link_id)

            # travel time
            travel_time = self.getSumTravelTimeOfLink(link_id)
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

