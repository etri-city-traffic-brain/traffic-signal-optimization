
import os
import torch
from xml.etree.ElementTree import parse
from gen_net import Network
from configs import EXP_CONFIGS


class MapNetwork(Network):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.tl_rl_list = list()
        self.offset_list = list()
        self.phase_list = list()
        self.common_phase = list()
        self.net_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.net.xml')
        self.rou_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.rou.xml')

    def get_tl_from_add_xml(self):
        add_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.add.xml')
        NET_CONFIGS = dict()
        NET_CONFIGS['phase_num_actions'] = {2: [[0, 0], [1, -1]],
                                            3: [[0, 0, 0], [1, 0, -1], [1, -1, 0], [0, 1, -1], [-1, 0, 1], [0, -1, 1], [-1, 1, 0]],
                                            4: [[0, 0, 0, 0], [1, 0, 0, -1], [1, 0, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1],
                                                [1, 0, 0, -1], [1, 0, -1, 0], [1, 0, 0, -1], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1], [1, 1, -1, -1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1], [-1, 1, -1, 1]],
                                            5: [[0, 0, 0, 0, 0]],
                                            6: [[0, 0, 0, 0, 0, 0]], }
        NET_CONFIGS['rate_action_space'] = dict()
        for i in range(2, 7):  # rate action_space 지정
            NET_CONFIGS['rate_action_space'][i] = len(
                NET_CONFIGS['phase_num_actions'][i])

        NET_CONFIGS['tl_period'] = list()
        traffic_info = dict()
        add_net_tree = parse(add_file_path)
        tlLogicList = add_net_tree.findall('tlLogic')
        NET_CONFIGS['time_action_space'] = list()

        # traffic info 저장
        for tlLogic in tlLogicList:
            tl_id = tlLogic.attrib['id']
            traffic_info[tl_id] = dict()
            traffic_node_info = traffic_info[tl_id]
            traffic_node_info['min_phase'] = list()
            traffic_node_info['phase_duration'] = list()
            traffic_node_info['max_phase'] = list()
            traffic_node_info['min_phase'] = list()
            traffic_node_info['min_phase'] = list()

            # rl agent 갯수 정리
            self.tl_rl_list.append(tlLogic.attrib['id'])  # rl 조종할 tl_rl추가
            # offset 저장
            traffic_node_info['offset'] = int(tlLogic.attrib['offset'])
            self.offset_list.append(traffic_node_info['offset'])

            # phase전체 찾기
            phaseList = tlLogic.findall('phase')
            phase_state_list = list()
            phase_duration_list = list()
            common_phase_list = list()
            phase_index_list = list()
            min_duration_list = list()
            max_duration_list = list()
            tl_period = 0  # phase set의 전체 길이
            # 각 phase에 대해서 길이 찾기 등등
            num_phase = 0  # phase갯수 filtering
            for i, phase in enumerate(phaseList):
                phase_state_list.append(phase.attrib['state'])
                phase_duration_list.append(int(phase.attrib['duration']))
                tl_period += int(phase.attrib['duration'])
                if int(phase.attrib['duration']) > 5:  # Phase 로 간주할 숫자
                    num_phase += 1
                    min_duration_list.append(int(phase.attrib['minDur']))
                    max_duration_list.append(int(phase.attrib['maxDur']))
                    phase_index_list.append(i)
                    common_phase_list.append(int(phase.attrib['duration']))

            # dictionary에 담기
            traffic_node_info['phase_list'] = phase_state_list
            traffic_node_info['phase_duration'] = phase_duration_list
            traffic_node_info['common_phase'] = common_phase_list
            traffic_node_info['phase_index'] = phase_index_list
            # 각 신호별 길이
            traffic_node_info['period'] = tl_period
            NET_CONFIGS['tl_period'].append(tl_period)
            traffic_node_info['matrix_actions'] = NET_CONFIGS['phase_num_actions'][num_phase]
            traffic_node_info['min_phase'] = min_duration_list
            traffic_node_info['max_phase'] = max_duration_list
            traffic_node_info['num_phase'] = num_phase
            # 각 tl_rl의 time_action_space지정
            # NET_CONFIGS['time_action_space'].append(abs(round((torch.min(torch.tensor(traffic_node_info['max_phase'])-torch.tensor(
            #     traffic_node_info['common_phase']), torch.tensor(traffic_node_info['common_phase'])-torch.tensor(traffic_node_info['min_phase']))/2).mean().item())))
            NET_CONFIGS['time_action_space'].append(4)  # 임의 초 지정

            self.phase_list.append(phase_state_list)
            self.common_phase.append(phase_duration_list)

        # TODO  node interest pair 계산기 network base에 생성
        maximum = 0
        for key in traffic_info.keys():
            if maximum < len(traffic_info[key]['phase_duration']):
                maximum = len(traffic_info[key]['phase_duration'])
        NET_CONFIGS['max_phase_num'] = maximum

        # road용
        # edge info 저장
        self.configs['edge_info'] = list()
        edge_list = list()  # edge존재 확인용
        net_tree = parse(self.net_file_path)
        edges = net_tree.findall('edge')
        for edge in edges:
            if 'function' not in edge.attrib.keys():
                edge_list.append({
                    'id': edge.attrib['id'],
                    'from': edge.attrib['from'],
                    'to': edge.attrib['to'],
                })
            self.configs['edge_info'].append(edge.attrib['id'])  # 모든 엣지 저장
        # node info 저장
        self.configs['node_info'] = list()
        node_list = list()
        # interest list
        interest_list = list()
        # node interest pair
        node_interest_pair = dict()
        junctions = net_tree.findall('junction')
        # state space size 결정
        inflow_size = 0
        # network용
        for junction in junctions:
            node_id = junction.attrib['id']
            if junction.attrib['type'] == "traffic_light":  # 정상 node만 분리, 신호등 노드
                node_list.append({
                    'id': node_id,
                    'type': junction.attrib['type'],
                })
                if node_id in self.tl_rl_list:  # 학습하는 tl만 저장
                    i = 0
                    interests = list()
                    for edge in edge_list:
                        interest = dict()
                        if edge['to'] == node_id:  # inflow
                            interest['id'] = node_id+'_{}'.format(i)
                            interest['inflow'] = edge['id']
                            for target_edge in edge_list:
                                if target_edge['from'] == edge['to'] and target_edge['to'] == edge['from']:
                                    interest['outflow'] = target_edge['id']
                                    break
                                else:
                                    interest['outflow'] = None

                            interests.append(interest)
                            i += 1  # index표기용

                        elif edge['from'] == node_id:
                            interest['id'] = node_id+'_{}'.format(i)
                            interest['outflow'] = edge['id']
                            for target_edge in edge_list:
                                if target_edge['from'] == edge['to'] and target_edge['to'] == edge['from']:
                                    interest['inflow'] = target_edge['id']
                                    break
                                else:
                                    interest['inflow'] = None
                            interests.append(interest)
                            i += 1  # index표기용

                    # 중복이 존재하는지 확인 후 list에 삽입
                    no_dup_outflow_list = list()
                    no_dup_interest_list = list()
                    for interest_comp in interests:
                        if interest_comp['outflow'] not in no_dup_outflow_list:
                            no_dup_outflow_list.append(
                                interest_comp['outflow'])
                            no_dup_interest_list.append(interest_comp)
                    interest_list.append(no_dup_interest_list)
                    node_interest_pair[node_id] = no_dup_interest_list
                    if inflow_size < len(no_dup_interest_list):
                        inflow_size = len(no_dup_interest_list)

            # 일반 노드
            elif junction.attrib['type'] == "priority":  # 정상 node만 분리
                node_list.append({
                    'id': node_id,
                    'type': junction.attrib['type'],
                })
            else:
                pass
            self.configs['node_info'].append({
                'id': node_id,
                'type': junction.attrib['type'],
            })
            # 정리
        NET_CONFIGS['edge_info'] = self.configs['edge_info']
        NET_CONFIGS['node_info'] = self.configs['node_info']
        NET_CONFIGS['traffic_node_info'] = traffic_info
        NET_CONFIGS['interest_list'] = interest_list
        NET_CONFIGS['node_interest_pair'] = node_interest_pair
        NET_CONFIGS['tl_rl_list'] = self.tl_rl_list
        NET_CONFIGS['offset'] = self.offset_list
        NET_CONFIGS['phase_list'] = self.phase_list
        NET_CONFIGS['common_phase'] = self.common_phase
        NET_CONFIGS['state_space'] = inflow_size*2  # 좌회전,직전
        print("Agent Num:{}, Traffic Num:{}".format(
            len(self.tl_rl_list), len(node_list)))
        return NET_CONFIGS

    def get_tl_from_xml(self):
        if os.path.exists(os.path.join(self.configs['current_path'], 'Network', self.configs['load_file_name']+'.add.xml')):
            print("additional exists")
            return self.get_tl_from_add_xml()
        else:
            NET_CONFIGS = dict()
            NET_CONFIGS['phase_num_actions'] = {2: [[0, 0], [1, -1]],
                                                3: [[0, 0, 0], [1, 0, -1], [1, -1, 0], [0, 1, -1], [-1, 0, 1], [0, -1, 1], [-1, 1, 0]],
                                                4: [[0, 0, 0, 0], [1, 0, 0, -1], [1, 0, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1],
                                                    [1, 0, 0, -1], [1, 0, -1, 0], [1, 0, 0, -1], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1], [1, 1, -1, -1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1], [-1, 1, -1, 1]],
                                                5: [[0, 0, 0, 0, 0]],
                                                6: [[0, 0, 0, 0, 0, 0]], }
            NET_CONFIGS['rate_action_space'] = dict()
            for i in range(2, 7):  # rate action_space 지정
                NET_CONFIGS['rate_action_space'][i] = len(
                    NET_CONFIGS['phase_num_actions'][i])

            NET_CONFIGS['tl_period'] = list()
            traffic_info = dict()
            net_tree = parse(self.net_file_path)
            tlLogicList = net_tree.findall('tlLogic')
            NET_CONFIGS['time_action_space'] = list()

            # traffic info 저장
            for tlLogic in tlLogicList:
                tl_id = tlLogic.attrib['id']
                traffic_info[tl_id] = dict()
                traffic_node_info = traffic_info[tl_id]
                traffic_node_info['min_phase'] = list()
                traffic_node_info['phase_duration'] = list()
                traffic_node_info['max_phase'] = list()
                traffic_node_info['min_phase'] = list()
                traffic_node_info['min_phase'] = list()

                # rl agent 갯수 정리
                self.tl_rl_list.append(tlLogic.attrib['id'])  # rl 조종할 tl_rl추가
                # offset 저장
                traffic_node_info['offset'] = int(tlLogic.attrib['offset'])
                self.offset_list.append(traffic_node_info['offset'])

                # phase전체 찾기
                phaseList = tlLogic.findall('phase')
                phase_state_list = list()
                phase_duration_list = list()
                common_phase_list = list()
                phase_index_list = list()
                min_duration_list = list()
                max_duration_list = list()
                tl_period = 0  # phase set의 전체 길이
                # 각 phase에 대해서 길이 찾기 등등
                num_phase = 0  # phase갯수 filtering
                for i, phase in enumerate(phaseList):
                    phase_state_list.append(phase.attrib['state'])
                    this_phase_dur = phase.attrib['duration']
                    phase_duration_list.append(int(this_phase_dur))
                    tl_period += int(this_phase_dur)
                    # Phase 로 간주할 숫자
                    if int(this_phase_dur) > 5 and 'minDur' in phase.attrib.keys() and 'maxDur' in phase.attrib.keys():
                        num_phase += 1
                        min_duration_list.append(
                            int(phase.attrib['minDur']))
                        max_duration_list.append(
                            int(phase.attrib['maxDur']))
                        phase_index_list.append(i)
                        common_phase_list.append(int(this_phase_dur))
                    elif int(this_phase_dur) > 5:
                        num_phase += 1
                        min_duration_list.append(
                            int(this_phase_dur)-5)
                        max_duration_list.append(
                            int(this_phase_dur)+5)
                        phase_index_list.append(i)
                        common_phase_list.append(int(this_phase_dur))

                # dictionary에 담기
                traffic_node_info['phase_list'] = phase_state_list
                traffic_node_info['phase_duration'] = phase_duration_list
                traffic_node_info['common_phase'] = common_phase_list
                traffic_node_info['phase_index'] = phase_index_list
                # 각 신호별 길이
                traffic_node_info['period'] = tl_period
                NET_CONFIGS['tl_period'].append(tl_period)
                traffic_node_info['matrix_actions'] = NET_CONFIGS['phase_num_actions'][num_phase]
                traffic_node_info['min_phase'] = min_duration_list
                traffic_node_info['max_phase'] = max_duration_list
                traffic_node_info['num_phase'] = num_phase
                # 각 tl_rl의 time_action_space지정
                NET_CONFIGS['time_action_space'].append(abs(round((torch.min(torch.tensor(traffic_node_info['max_phase'])-torch.tensor(
                    traffic_node_info['common_phase']), torch.tensor(traffic_node_info['common_phase'])-torch.tensor(traffic_node_info['min_phase'])).float()).mean().item())))

                self.phase_list.append(phase_state_list)
                self.common_phase.append(phase_duration_list)

            # TODO  node interest pair 계산기 network base에 생성
            maximum = 0
            for key in traffic_info.keys():
                if maximum < len(traffic_info[key]['phase_duration']):
                    maximum = len(traffic_info[key]['phase_duration'])
            NET_CONFIGS['max_phase_num'] = maximum

            # road용
            # edge info 저장
            self.configs['edge_info'] = list()
            edges = net_tree.findall('edge')
            for edge in edges:
                if 'function' not in edge.attrib.keys():
                    self.configs['edge_info'].append({
                        'id': edge.attrib['id'],
                        'from': edge.attrib['from'],
                        'to': edge.attrib['to'],
                    })
            # node info 저장
            self.configs['node_info'] = list()
            node_list = list()
            # interest list
            interest_list = list()
            # node interest pair
            node_interest_pair = dict()
            junctions = net_tree.findall('junction')
            # state space size 결정
            inflow_size = 0
            # network용
            for junction in junctions:
                node_id = junction.attrib['id']
                if junction.attrib['type'] == "traffic_light":  # 정상 node만 분리, 신호등 노드
                    node_list.append({
                        'id': node_id,
                        'type': junction.attrib['type'],
                    })
                    # node 결정 완료
                    # edge는?
                    i = 0
                    interests = list()
                    for edge in self.configs['edge_info']:
                        interest = dict()
                        if edge['to'] == node_id:  # inflow
                            interest['id'] = node_id+'_{}'.format(i)
                            interest['inflow'] = edge['id']
                            for tmpEdge in self.configs['edge_info']:  # outflow
                                if tmpEdge['from'] == node_id and edge['from'] == tmpEdge['to']:
                                    interest['outflow'] = tmpEdge['id']
                                    break
                                else:
                                    interest['outflow'] = None
                            # tmp_edge=str(-int(edge['id']))
                            # if tmp_edge in edge_list:
                            #     interest['outflow']=tmp_edge
                            # else:
                            #     interest['outflow']=None
                            interests.append(interest)
                            i += 1  # index표기용

                        elif edge['from'] == node_id:
                            interest['id'] = node_id+'_{}'.format(i)
                            interest['outflow'] = edge['id']
                            for tmpEdge in self.configs['edge_info']:  # outflow
                                if tmpEdge['to'] == node_id and edge['to'] == tmpEdge['from']:
                                    interest['inflow'] = tmpEdge['id']
                                    break
                                else:
                                    interest['inflow'] = None
                            # tmp_edge=str(-int(edge['id']))
                            # if tmp_edge in edge_list:
                            #     interest['inflow']=tmp_edge
                            # else:
                            #     interest['inflow']=None
                            interests.append(interest)
                            i += 1  # index표기용

                    # 중복이 존재하는지 확인 후 list에 삽입
                    no_dup_outflow_list = list()
                    no_dup_interest_list = list()
                    for interest_comp in interests:
                        if interest_comp['outflow'] not in no_dup_outflow_list:
                            no_dup_outflow_list.append(
                                interest_comp['outflow'])
                            no_dup_interest_list.append(interest_comp)
                    interest_list.append(no_dup_interest_list)
                    node_interest_pair[node_id] = no_dup_interest_list
                    if inflow_size < len(no_dup_interest_list):
                        inflow_size = len(no_dup_interest_list)

                # 일반 노드
                elif junction.attrib['type'] == "priority":  # 정상 node만 분리
                    node_list.append({
                        'id': node_id,
                        'type': junction.attrib['type'],
                    })
                else:
                    pass
                self.configs['node_info'].append({
                    'id': node_id,
                    'type': junction.attrib['type'],
                })

            # 정리
            NET_CONFIGS['node_info'] = self.configs['node_info']
            NET_CONFIGS['edge_info'] = self.configs['edge_info']

            NET_CONFIGS['traffic_node_info'] = traffic_info
            NET_CONFIGS['interest_list'] = interest_list
            NET_CONFIGS['node_interest_pair'] = node_interest_pair
            NET_CONFIGS['tl_rl_list'] = self.tl_rl_list
            NET_CONFIGS['offset'] = self.offset_list
            NET_CONFIGS['phase_list'] = self.phase_list
            NET_CONFIGS['common_phase'] = self.common_phase
            NET_CONFIGS['state_space'] = inflow_size*2  # 좌회전,직전

            return NET_CONFIGS

    def gen_net_from_xml(self):
        net_tree = parse(self.net_file_path)
        if self.configs['mode'] == 'train' or self.configs['mode'] == 'test':
            gen_file_name = str(os.path.join(self.configs['current_path'], 'training_data',
                                             self.configs['time_data'], 'net_data', self.configs['time_data']+'.net.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8',
                           xml_declaration=True)
        else:  # simulate
            gen_file_name = str(os.path.join(
                self.configs['current_path'], 'Net_data', self.configs['time_data']+'.net.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8',
                           xml_declaration=True)

    def gen_rou_from_xml(self):
        net_tree = parse(self.rou_file_path)
        if self.configs['mode'] == 'train' or self.configs['mode'] == 'test':
            gen_file_name = str(os.path.join(self.configs['current_path'], 'training_data',
                                             self.configs['time_data'], 'net_data', self.configs['time_data']+'.rou.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8',
                           xml_declaration=True)
        else:
            gen_file_name = str(os.path.join(self.configs['current_path'], 'Net_data',
                                             self.configs['time_data']+'.rou.xml'))
            net_tree.write(gen_file_name, encoding='UTF-8',
                           xml_declaration=True)
