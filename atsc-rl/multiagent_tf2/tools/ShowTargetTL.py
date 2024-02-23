# -*- coding: utf-8 -*-
#
# Show training target TLs
#
#  [$] conda activate UniqOpt.p3.8.v2
#  [$] python ./tools/ShowTargetTL.py   --scenario-file-path data/envs/salt --map dj200
#        --target-TL  "SA 3,SA 28,SA 101,SA 6,SA 41,SA 20,SA 37,SA 38,SA 9,SA 1,SA 57,SA 102,SA 104,SA 98,SA 8,SA 33,SA 59,SA 30"

#
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # "0" # "0,1,2"
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

from env.off_ppo.SaltEnvUtil import getSaRelatedInfo
from env.off_ppo.SaltEnvUtil import makePosssibleSaNameList
from env.off_ppo.SaltEnvUtil import getScenarioRelatedFilePath, constructTSSRelatedInfo
from TSOUtil import makePosssibleSaNameList
from TSOUtil import addArgumentsToParser
from TSOUtil import removeWhitespaceBtnComma


def parseArgument():
    '''
    argument parsing
    :return:
    '''

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

def prettyList(list_data, num_item_per_line):

    cnt = 0
    buf=""
    for item in list_data:
        if cnt % num_item_per_line == 0:
            buf=f"{buf}\t"
        else:
            buf = f"{buf},"
        cnt += 1
        buf = f"{buf} {item}"
        if cnt % num_item_per_line == 0:
            buf=f"{buf}\n"

    return buf


def showTargetTLs():
    '''
    show target TLs to optimize
    :param args:
    :return:
    '''

    args = parseArgument()
    args.target_TL = removeWhitespaceBtnComma(args.target_TL)

    #print(f"\n\n\n")
    print(f"============================================================================")
    # print("##### showTargetTLsV2")
    # print(f"target Sub-Areas : {args.target_TL}")
    _target_sa_list = args.target_TL.split(",")
    cvted_list = prettyList(_target_sa_list, 5)
    print(f"target Sub-Areas : {len(_target_sa_list)} SAs")
    print(cvted_list)

    print(f"----------------------------------------------------------------------------")

    # 1. gather information from TSS file
    possible_sa_name_list = makePosssibleSaNameList(args.target_TL)
    # print(f"possible_sa_name_list={possible_sa_name_list}")
    salt_scenario = args.scenario_file_path
    _, _, edge_file_path, tss_file_path = getScenarioRelatedFilePath(args.scenario_file_path)
    target_tl_obj = constructTSSRelatedInfo(args, tss_file_path, possible_sa_name_list)

    target_sa_obj = {}
    for tl_obj in target_tl_obj:
        if target_tl_obj[tl_obj]['signalGroup'] not in target_sa_obj:
            target_sa_obj[target_tl_obj[tl_obj]['signalGroup']] = {}
            target_sa_obj[target_tl_obj[tl_obj]['signalGroup']]['crossName_list'] = []  # 교차로 그룹에 속한 교차로 이름 목록
            target_sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tlid_list'] = []  # 교차로 id 리스트
        target_sa_obj[target_tl_obj[tl_obj]['signalGroup']]['crossName_list'].append(target_tl_obj[tl_obj]['crossName'])
        target_sa_obj[target_tl_obj[tl_obj]['signalGroup']]['tlid_list'].append(tl_obj)

    target_sa_name_list = list(target_sa_obj.keys())
    target_tl_id_list = list(target_tl_obj.keys())

    # 2. show gathered information
    cnt_sa = len(target_sa_name_list)
    total_num_TLs = 0
    target_sa_name_list.sort()
    for sa_name in target_sa_name_list:
        cross_name_list = target_sa_obj[sa_name]['crossName_list']
        num_TLs = len(cross_name_list)
        # print(f"{sa_name}({num_TLs} Traffic Lights) : {cross_name_list}")
        print(f"{sa_name} : {num_TLs} TLs")
        # print(f"\t{cross_name_list}")
        cvted_list = prettyList(cross_name_list, 5)
        print(cvted_list)
        total_num_TLs += num_TLs

    print(f"----------------------------------------------------------------------------")
    print(f"\tTraining Target : {total_num_TLs} Traffic Lights within {cnt_sa} SAs ")
    print(f"----------------------------------------------------------------------------")
    print(f" We are currently training RL agents to control traffic lights.")
    print(f" Agents are controlling traffic lights using trained models simultaneously.")
    print(f"============================================================================")
    #print(f"\n\n\n")



if __name__ == "__main__":
    ## dump launched time
    #launched = datetime.datetime.now()
    #print(f'TSO(pid={os.getpid()}) launched at {launched}')

    showTargetTLs()
