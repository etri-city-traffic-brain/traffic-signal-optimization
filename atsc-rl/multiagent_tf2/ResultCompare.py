import pandas as pd
import numpy as np

import pprint
from DebugConfiguration import DBG_OPTIONS
from TSOConstants import RESULT_COMPARE_SKIP


def processStatisticalInformation(field, op, op2, ft_0, ft_all, rl_0, rl_all, individual_output):
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

    #todo should care when ft_passed is 0
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

    #todo should care when ft_passed is 0
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



def getStatisticsInformationAboutGivenEdgeList(ft_output, rl_output, in_edge_list_0, in_edge_list, cut_interval):
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



def compareResult(args, target_tl_obj, ft_output, rl_output, model_num):
    '''
    :param args:
    :param target_tl_obj: information about target TL
    :param ft_output: a data frame object which was generated by reading an output (csv) file of simulator
                           that performed the signal control simulation based on the fixed signal
    :param rl_output: a data frame object which was generated by reading an output (csv) file of simulator
                           that performed signal control simulation based on reinforcement learning inference
    :param model_num: number which indicate optimal model which was used to TEST
    :return:
    '''

    # print("target_tl_obj")
    # pprint.pprint(target_tl_obj, width=200, compact=True)

    ##
    ## Various statistical information related to intersections is extracted from the DataFrame object
    ##      containing the contents of the CSV file created by the simulator.
    ##-- create empty DataFrame object
    total_output = pd.DataFrame()

    cut_interval = args.start_time + RESULT_COMPARE_SKIP # 2시간 테스트시 앞에 1시간은 비교대상에서 제외
    print(f"training step: {args.start_time} to {args.end_time}")
    print(f"comparing step: {cut_interval} to {args.end_time}")
    print(f"model number: {model_num}")

    ##-- set the info to be extracted : kind, method
    ##---- kinds of information to be extracted
    varList = ['VehPassed', 'AverageSpeed', 'WaitingTime', 'AverageDensity', 'SumTravelTime', 'WaitingQLength']

    ##----methods how to calculate
    ##     larger is good if this value is positive, smaller is good if this value is negative
    varOp = [1, 1, -1, -1, -1, -1]
    varOp2 = ['sum', 'mean', 'sum', 'mean', 'sum', 'mean']

    ##-- traverse intersection and process statistical info for each intersection
    for tl in target_tl_obj:
        # Change the signalGroup name to the same format : ex, 101 --> SA 101
        if "SA " not in target_tl_obj[tl]['signalGroup']:
            target_tl_obj[tl]['signalGroup'] = 'SA ' + target_tl_obj[tl]['signalGroup']

        # add columns : crossName, signalGroup
        individual_output = pd.DataFrame(
            {'name': [target_tl_obj[tl]['crossName']], 'SA': [target_tl_obj[tl]['signalGroup']]})

        # gather incomming edge info
        in_edge_list = []
        in_edge_list_0 = []
        in_edge_list = np.append(in_edge_list, target_tl_obj[tl]['in_edge_list'])
        in_edge_list_0 = np.append(in_edge_list_0, target_tl_obj[tl]['in_edge_list_0'])

        if DBG_OPTIONS.PrintResultCompare:
            print(target_tl_obj[tl]['crossName'], target_tl_obj[tl]['in_edge_list_0'])

        ft_output2, ft_output3, rl_output2, rl_output3 = getStatisticsInformationAboutGivenEdgeList(ft_output, rl_output, in_edge_list_0, in_edge_list, cut_interval)

        # process by information type(kind) and add it to DataFrame object
        for v in range(len(varList)):
            individual_output = processStatisticalInformation(varList[v], varOp[v], varOp2[v],
                                          ft_output2, ft_output3, rl_output2, rl_output3, individual_output)

        total_output = pd.concat([total_output, individual_output])

    in_edge_list = []
    in_edge_list_0 = []

    ##
    ## process statistical info about entire intersection
    individual_output = pd.DataFrame({'name': ['total']})

    ##-- construct EDGE info about entire intersection
    for tl in target_tl_obj:
        in_edge_list = np.append(in_edge_list, target_tl_obj[tl]['in_edge_list'])
        in_edge_list_0 = np.append(in_edge_list_0, target_tl_obj[tl]['in_edge_list_0'])
        # if DBG_OPTIONS.PrintResultCompare:
        #     print(target_tl_obj[tl]['crossName'], target_tl_obj[tl]['in_edge_list_0'])

    if DBG_OPTIONS.PrintResultCompare:
        print("\nAll Target TL summary.....")

    ft_output2, ft_output3, rl_output2, rl_output3 = getStatisticsInformationAboutGivenEdgeList(ft_output, rl_output, in_edge_list_0, in_edge_list, cut_interval)

    # process by information type(kind) and add it to DataFrame object
    for v in range(len(varList)):
        individual_output = processStatisticalInformation(varList[v], varOp[v], varOp2[v],
                                                          ft_output2, ft_output3, rl_output2, rl_output3,
                                                          individual_output)

    total_output = pd.concat([total_output, individual_output])

    total_output = total_output.sort_values(by=["SA"], ascending=True)

    return total_output

