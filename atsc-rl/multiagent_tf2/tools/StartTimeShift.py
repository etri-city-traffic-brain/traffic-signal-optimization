# -*- coding: utf-8 -*-
import json
import shutil
import xml.etree.ElementTree as ET

DISTRIBUTE=False
MAGIC_SKIP = 60 # 60 sec

def startTimeShift(src_path, dst_path, f_name, op, shift):
    '''
    shift start time

    :param src_path: file path of route file to be shifted
    :param dst_path: file path to save converted route file
    :param f_name: route file name
    :param shift: seconds to shift
    :return:
    '''
    print("shift={} sec".format(shift))
    tree = ET.parse(f"{src_path}/{f_name}")
    root = tree.getroot()
    vehicles = root.findall("vehicle")

    for x in vehicles:
        x.attrib["depart"] = str(float(x.attrib["depart"]) + shift)

    tokens=f_name.split(".")
    fn=('.').join(tokens[:-1])

    if DISTRIBUTE:
        cvted_file_name = f"{fn}.test.{tokens[-1]}"
    else:
        cvted_file_name = f"{fn}.{op}_{abs(shift)}.{tokens[-1]}"

    tree.write(f"{dst_path}/{cvted_file_name}")
    return cvted_file_name


def createScenarioFile(src_path, dst_path, mode,  map_name, route_fn, shift):
    '''
    generate scenario file with changed start/end time and route file

    :param src_path: file path of route file to be shifted
    :param dst_path: file path to save converted route file
    :param mode:
    :param map_name : name of map
    :param route_fn: route file name
    :param shift: seconds to shift
    :return:
    '''
    fn_src = f'{src_path}/{map_name}_{mode}.scenario.json'

    with open(fn_src, "r") as json_file:
        scenario = json.load(json_file)
        if DISTRIBUTE:
            scenario["scenario"]["time"]["end"] = int(scenario["scenario"]["time"]["end"]) - MAGIC_SKIP
        else:
            scenario["scenario"]["time"]["begin"] = int(scenario["scenario"]["time"]["begin"]) + shift
            scenario["scenario"]["time"]["end"] = int(scenario["scenario"]["time"]["end"]) + shift
        scenario["scenario"]["input"]["route"] = route_fn
        # print(json.dumps(scenario, indent=4))

    fn_dst = f'{dst_path}/{map_name}_{mode}.scenario.json'
    with open(fn_dst, "w") as json_file:
        json.dump(scenario, json_file, indent=2)


def createStartTimeShiftedScenario(src_path, route_file, shift_length_list):
    for shift in shift_length_list:
        print(f' create new scenario file from {src_path} by shifting start time {shift} sec ')

        if shift >= 0:
            op = "inc"
        else:
            op = "dec"

        dst_path = f'{src_path}.{op}.{abs(shift)}'

        # os.makedirs(dst_path, exist_ok=True)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

        ##-- create shifted route file
        cvted_route_fn = startTimeShift(src_path, dst_path, route_file, op, int(shift))

        ##-- create scenario file for fixed time-based control
        tokens = route_file.split(".")
        map_name = tokens[0]
        mode = "simulate"
        createScenarioFile(src_path, dst_path, mode, map_name, cvted_route_fn, shift)

        ##-- create scenario file for RL agent-based control
        mode = "test"
        createScenarioFile(src_path, dst_path, mode, map_name, cvted_route_fn, shift)



if __name__ == "__main__":
    if 0:
        testStartTimeShiftOrgSucc()

    file_path = "./salt/doan"
    route_file = "doan.rou.xml"
    shift_length_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 150, 300, 600]
    #shift_length_list = [5, -5, -25]
    #shift_length_list = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -70, -80, -90, -100, -150, -300, -600]
    createStartTimeShiftedScenario(file_path, route_file, shift_length_list)
