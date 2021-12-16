import argparse
import json
import os
import sys
import time
from xml.etree.ElementTree import parse
from torch.utils.tensorboard import SummaryWriter


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compare with file_a and file_b",
        epilog="python run.py file_a file_b")

    # required input parameters
    parser.add_argument(
        'file_a', type=str,
        help='control group file name')

    # optional input parameters
    parser.add_argument(
        'file_b', type=str,
        help='comparision group file name')
    parser.add_argument(
        '--type', type=str, default='edge',
        help='choose comparing type lane or edge')
    parser.add_argument(
        '--data', type=str, default='speed',
        help='choose data you want to see above "speed","density","waitingTime"and "occupancy"')
    return parser.parse_known_args(args)[0]


def main(args):
    flags = parse_args(args)
    showData = flags.data
    a_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'data', flags.file_a)
    b_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'data', flags.file_b)
    if os.path.exists(a_path) == False or os.path.exists(b_path) == False:
        raise FileNotFoundError
    time_str = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    tree = list()
    root = list()
    writer = list()
    interval = list()
    writer.append(SummaryWriter(os.path.join(
        'tensorboard', flags.type, time_str, flags.file_a)))
    writer.append(SummaryWriter(os.path.join(
        'tensorboard', flags.type, time_str, flags.file_b)))
    tree.append(parse(a_path))
    tree.append(parse(b_path))
    for idx in range(2):  # 비교군 대조군
        end_time = list()
        root.append(tree[idx].getroot())
        interval.append(root[idx].findall('interval'))
        for i, _ in enumerate(interval[idx]):
            end_time.append(float(interval[idx][i].attrib['end']))

        if flags.type == 'edge':
            for i, t in enumerate(end_time):
                id_list = [[], []]
                speed_list = [[], []]
                edge_list = [[], []]
                edge_list[idx] = interval[idx][i].findall('edge')
                for _, edge in enumerate(edge_list[idx]):
                    id_list[idx].append(edge.attrib['id'])
                    if showData in edge.keys():
                        speed_list[idx].append(float(edge.attrib[showData]))
                    else:
                        speed_list[idx].append(float('0.0'))

                for j, id in enumerate(id_list[idx]):
                    writer[idx].add_scalar(
                        '{}/{}'.format(showData, id), speed_list[idx][j], t)
                writer[idx].flush()

        elif flags.type == 'lane':
            for i, t in enumerate(end_time):
                id_list = [[], []]
                speed_list = [[], []]
                edge_list = [[], []]
                edge_list[idx] = interval[idx][i].findall('edge')
                for _, edge in enumerate(edge_list[idx]):
                    id_list[idx].append(edge.attrib['id'])
                    lane_list = edge.findall('lane')
                    speed_dict = dict()
                    for _, lane in enumerate(lane_list):
                        if showData in lane.keys():
                            speed_dict[lane.attrib['id'][-1]
                                       ] = float(lane.attrib[showData])
                        else:
                            speed_dict[lane.attrib['id'][-1]] = float('0.0')
                        speed_list[idx].append(speed_dict)

                for j, id in enumerate(id_list[idx]):
                    for k in range(len(speed_dict.keys())):
                        writer[idx].add_scalar(
                            '{}/{}/{}'.format(showData, id, k), speed_list[idx][j][str(k)], t)
                writer[idx].flush()

    writer[0].close()
    writer[1].close()


if __name__ == '__main__':
    main(sys.argv[1:])
