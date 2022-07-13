import pandas as pd
import numpy as np

import pprint
from DebugConfiguration import DBG_OPTIONS
from TSOConstants import _RESULT_COMPARE_SKIP_



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



def compareResult(args, target_tl_obj, ft_output, rl_output, model_num, passed_res_comp_skip = -1):
    '''
    compare two result files and calculate improvement rate

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

        #todo ft_output과 rl_output에  AvgTravelTime 추가
        #   AvgTravelTime = SumTravelTime / VehPassed
        #     ['VehPassed', 'AverageSpeed', 'WaitingTime', 'AverageDensity', 'SumTravelTime', 'AvgTravelTime', 'WaitingQLength']

        #     frame['avg']=frame['kor']/frame['n']
        # todo NaN을 0으로 바꾸기, inf를 0으로 바꾸기
        ft_output['AvgTravelTime'] = ft_output['SumTravelTime'] / ft_output['VehPassed']
        for i in ft_output[ft_output['VehPassed']==0.0].index:
            ft_output.at[i, 'AvgTravelTime']=0.0

        rl_output['AvgTravelTime'] = rl_output['SumTravelTime'] / rl_output['VehPassed']
        for i in rl_output[rl_output['VehPassed']==0.0].index:
            rl_output.at[i, 'AvgTravelTime']=0.0

    return __compareResult(args, target_tl_obj, ft_output, rl_output, model_num, passed_res_comp_skip)



def __compareResult(args, target_tl_obj, ft_output, rl_output, model_num, passed_res_comp_skip = -1):
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
    target_sa_tl_dic = {} # to save TL info per SA
    for tl in target_tl_obj:
        if "SA " not in target_tl_obj[tl]['signalGroup']:
            target_tl_obj[tl]['signalGroup'] = 'SA ' + target_tl_obj[tl]['signalGroup']
            # add columns : crossName, signalGroup
        individual_output = pd.DataFrame(
            {'name': [target_tl_obj[tl]['crossName']], 'SA': [target_tl_obj[tl]['signalGroup']]})

        individual_output = __compareResultInternal(individual_output, [tl], target_tl_obj, ft_output, rl_output, cut_interval)
        total_output = pd.concat([total_output, individual_output])

        ## gather SA info
        sa_name = target_tl_obj[tl]['signalGroup']
        if sa_name in target_sa_tl_dic.keys():
            target_sa_tl_dic[sa_name].append(tl)
        else:
            target_sa_tl_dic[sa_name] = [tl]

    #
    # for each SA
    3
    for sa in target_sa_tl_dic.keys():
        if DBG_OPTIONS.PrintResultCompare:
            print(f'{sa}')
            for tl in list(target_sa_tl_dic[sa]):
                print(target_tl_obj[tl]['crossName'])

        individual_output = pd.DataFrame({'name': [sa], 'SA': ['total']})
        individual_output = __compareResultInternal(individual_output, list(target_sa_tl_dic[sa]), target_tl_obj,
                                                    ft_output, rl_output, cut_interval)
        total_output = pd.concat([total_output, individual_output])

    #
    # for entire target
    #
    individual_output = pd.DataFrame({'name': ['total'], 'SA':['total']})
    individual_output = __compareResultInternal(individual_output, list(target_tl_obj.keys()), target_tl_obj, ft_output, rl_output, cut_interval)
    total_output = pd.concat([total_output, individual_output])

    total_output = total_output.sort_values(by=["SA", "name"], ascending=True)

    return total_output



def __compareResultInternal(individual_output, comp_tl_list, target_tl_obj, ft_output, rl_output, cut_interval):
    ##-- set the info to be extracted : kind, method
    ##---- kinds of information to be extracted
    if DBG_OPTIONS.IngCompResult:
        varList = ['VehPassed', 'AverageSpeed', 'WaitingTime', 'AverageDensity', 'SumTravelTime', 'AvgTravelTime', 'WaitingQLength']
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
        getStatisticsInformationAboutGivenEdgeList(ft_output, rl_output, in_edge_list_0, in_edge_list, cut_interval)

    # process by information type(kind) and add it to DataFrame object
    for v in range(len(varList)):
        individual_output = processStatisticalInformation(varList[v], varOp[v], varOp2[v],
                                                          ft_output2, ft_output3, rl_output2, rl_output3,
                                                          individual_output)
    return individual_output



def testCompareResult():
    import os
    from TSOConstants import _RESULT_COMP_
    import argparse
    # from run import parseArgument, createEnvironment
    from env.SaltEnvUtil import getSimulationStartStepAndEndStep
    from TSOUtil import makeConfigAndProblemVar
    from TSOUtil import addArgumentsToParser
    from env.SappoEnv import SaltSappoEnvV3

    def createEnvironment(args):
        env = -1
        if args.method == 'sappo':
            env = SaltSappoEnvV3(args)
        else:
            print("internal error : {} is not supported".format(args.method))

        return env



    def parseArgument():

        parser = argparse.ArgumentParser()

        parser = addArgumentsToParser(parser)

        args = parser.parse_args()

        args.scenario_file_path = f"{args.scenario_file_path}/{args.map}/{args.map}_{args.mode}.scenario.json"

        # todo : think how often should we update actions
        # if args.action == 'gr':
        #     args.control_cycle = 1

        # to use only exploitation when we do "test"
        if args.mode == 'test':
            args.epsilon = 0.0
            args.epsilon_min = 0.0

        return args


    if os.environ.get("UNIQ_OPT_HOME") is None:
        os.environ["UNIQ_OPT_HOME"] = os.getcwd()

    args = parseArgument()

    ## calculate trial length using argument and scenario file
    start_time, end_time = getSimulationStartStepAndEndStep(args)
    trial_len = end_time - start_time

    # set start_/end_time which will be used to test
    args.start_time = start_time
    args.end_time = end_time

    env = createEnvironment(args)

    ppo_config, problem_var = makeConfigAndProblemVar(args)

    ft_output = pd.read_csv("{}/output/simulate/{}".format(args.io_home, _RESULT_COMP_.SIMULATION_OUTPUT))
    rl_output = pd.read_csv("{}/output/test/{}".format(args.io_home, _RESULT_COMP_.SIMULATION_OUTPUT))

    total_output = compareResult(args, env.tl_obj, ft_output, rl_output, args.model_num, args.warmup_time)

    # result_fn = "{}/output/test/{}_{}.csv".format(args.io_home, problem_var, args.model_num)
    result_fn = "./zz.to_del.result_comp_test.csv"
    total_output.to_csv(result_fn, encoding='utf-8-sig', index=False)


    # get improve rate and dump it
    df = pd.read_csv(result_fn, index_col=0)
    improvement_rate = df.at[_RESULT_COMP_.ROW_NAME, _RESULT_COMP_.COLUMN_NAME]
    print("improvement_rate={} got from result comp file".format(improvement_rate))

    for sa in env.target_sa_name_list:
        __printImprovementRate(df, sa)
    __printImprovementRate(df, 'total')



def __printImprovementRate(df, target):
    ft_passed_num = df.at[target, 'ft_VehPassed_sum_0hop']
    rl_passed_num = df.at[target, 'rl_VehPassed_sum_0hop']
    ft_sum_travel_time = df.at[target, 'ft_SumTravelTime_sum_0hop']
    rl_sum_travel_time = df.at[target, 'rl_SumTravelTime_sum_0hop']

    ft_avg_travel_time = ft_sum_travel_time / ft_passed_num
    rl_avg_travel_time = rl_sum_travel_time / rl_passed_num
    imp_rate = (ft_avg_travel_time - rl_avg_travel_time) / ft_avg_travel_time * 100
    print(f'Average Travel Time ({target}): {imp_rate}% improved')

#
# python run.py --mode simulate --map doan --target-TL "SA 101, SA 104"
# python run.py --mode test --map doan --target-TL "SA 101, SA 104" --model-num 0
#
#  python ResultCompare.py --map doan --target-TL "SA 101, SA 104"

if __name__ == '__main__':
    DBG_OPTIONS.PrintResultCompare = True
    DBG_OPTIONS.IngCompResult = True
    testCompareResult()
