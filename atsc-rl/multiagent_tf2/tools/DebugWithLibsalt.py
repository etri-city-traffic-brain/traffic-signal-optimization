import libsalt
import numpy as np
import os


def increaseStep(inc):
    for i in range(inc):
        libsalt.simulationStep()


def doTestSpeed():
    inc = 10
    lane_id = "-563104834_1"
    increaseStep(inc)
    avg = libsalt.lane.getAverageSpeed(lane_id)
    avgVeh = libsalt.lane.getAverageVehicleSpeed(lane_id)
    curStep = libsalt.getCurrentStep()
    print(f"curStep={curStep} avgSpeed={avg}  avgVehSpeed={avgVeh}")


def doTestWT():
    inc = 100
    link_id = "-563104834"
    increaseStep(inc)
    curStep = libsalt.getCurrentStep()
    avg = libsalt.link.getAverageWaitingTime(link_id)
    avgVeh = libsalt.link.getAverageVehicleWaitingTime(link_id, curStep, curStep - 180)
    print(f"curStep={curStep} avgWT={avg}  avgVehWT={avgVeh}")

def foo(link_id, curStep, swichingTime):
    print(f"libsalt.getCurrentStep()={libsalt.getCurrentStep()}")
    print(
        f"libsalt.link.getAverageVehicleWaitingTime({link_id}, {curStep}, {swichingTime}) = {libsalt.link.getAverageVehicleWaitingTime(link_id, curStep, swichingTime)}")
    print(
        f"libsalt.link.getAverageVehicleWaitingQLength({link_id}, {curStep}, {swichingTime}) = {libsalt.link.getAverageVehicleWaitingQLength(link_id, curStep, swichingTime)}")
    print(
        f"libsalt.link.getNumWaitingVehicle({link_id}, {curStep}, {swichingTime}) = {libsalt.link.getNumWaitingVehicle(link_id, curStep, swichingTime)}")

    def changeTLPhase(tlid, logic_idx, duration_info):
        """
        change the TL logic indicated by given logic index(logic_idx) using the given duration info(duration_info)
        """
        import libsalt

        current_step = libsalt.getCurrentStep()
        current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
        current_phase_vector = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector

        if 1:
            print("0. In : phase duration={}".format(duration_info))
            print("1. get : current_phase_vector={}".format(current_phase_vector))
            # print("1. In : phase duration={}".format(duration_info), "current_phase_vector={}".format(current_phase_vector))

        # tuple은 수정할 수 없기에 list로 변경
        # ((32, 'rrgGgr'), (3, 'rryyyr'), ...) ==> [(32, 'rrgGgr'), (3, 'rryyyr'), ...]
        current_phase_vector = list(current_phase_vector)

        for i in range(len(duration_info)):
            # cur_logic.phases[i].duration = duration_info[i]
            #  ith phase에 대한 정보를 list로 변경
            ith_phase_info = list(current_phase_vector[i])
            ith_phase_info[0] = duration_info[i]

            current_phase_vector[i] = tuple(ith_phase_info)

        # list를 tuple로 변환
        phase_vector = tuple(current_phase_vector)

        if 1:
            print(
                "2. val  : current_step={}, tlid={}, current_schedule_id={}, phase_vector={}".format(current_step, tlid,
                                                                                                     current_schedule_id,
                                                                                                     phase_vector))
            print(
                "3. type : current_step={}, tlid={}, current_schedule_id={}, phase_vector={}".format(type(current_step),
                                                                                                     type(tlid), type(
                        current_schedule_id), type(phase_vector)))

        # libsalt.trafficsignal.setTLSPhaseVector(currentStep, tl_ID, scheduleID, phaseVector)
        libsalt.trafficsignal.setTLSPhaseVector(current_step, tlid, current_schedule_id, phase_vector)

        if 1:
            current_phase_vector = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector
            print("4. changed_phase_vector={}".format(current_phase_vector))

            raise Exception


if __name__ == "__main__":

    opt_home = "/home/tsoexp/PycharmProjects/traffic-signal-optimization-for-dist/atsc-rl/multiagent_tf2"
    scenario_file = "data/envs/salt/doan/doan_simulate.scenario.json"
    salt_scenario = f"{opt_home}/{scenario_file}"
    print(salt_scenario)
    libsalt.start(salt_scenario)

    tlid = "563103640"
    tlid = "cluster_563103641_563103889_563103894_563103895"  # 원골 네거리
    link_tuple = libsalt.trafficsignal.getTLSConnectedLinkID(tlid)  # ('-563104674', '-563104834', '563104746', '563104750')
    lane_list = ['-563104674_0', '-563104674_1',
                 '-563104834_0', '-563104834_1',
                 '563104746_0', '563104746_1', '563104746_2', '563104746_3',
                 '563104750_0', '563104750_1', '563104750_2', '563104750_3']
    link_id = link_tuple[1]
    lane_id = lane_list[3]

    for i in range(600):
        libsalt.simulationStep()

    if 1:
        current_step = libsalt.getCurrentStep()
        # to understand   __getGreenRatioAppliedPhaseArray(self, curr_sim_step, an_sa_obj, actions) at ActionMgmt
        current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
        mpv = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector
        mpv=list(mpv)
        print(mpv) # current phase vector
            ## [(26, 'rrrrrgrrGgrrrrrrGGG'), (4, 'rrrrryrryyrrrrrryyg'), (72, 'GGGGrrrrrrGGGGrrrrG'), (3, 'yyyyrrrrrryyyyrrrry'),
            #   (17, 'rrrrGgrrrgrrrrGrrrr'), (3, 'rrrryyrrryrrrryrrrr'), (51, 'rrrrrrGGrrrrrrrGrrr'), (4, 'rrrrrryyrrrrrrryrrr')]

        action = [1, 0, 0, 0, -1]
        green_idx = [0, 1, 2, 4, 6]
        add_time = 3

        for i in range(len(green_idx)):
            gi = green_idx[i]
            m = list(mpv[gi])
            m[0] = m[0] + int(action[i]) * add_time
            mpv[gi] = tuple(m)

        print(mpv)  # changed  phase vector
        ## [(29, 'rrrrrgrrGgrrrrrrGGG'), (4, 'rrrrryrryyrrrrrryyg'), (72, 'GGGGrrrrrrGGGGrrrrG'), (3, 'yyyyrrrrrryyyyrrrry'),
        #   (17, 'rrrrGgrrrgrrrrGrrrr'), (3, 'rrrryyrrryrrrryrrrr'), (48, 'rrrrrrGGrrrrrrrGrrr'), (4, 'rrrrrryyrrrrrrryrrr')]

        ### 변경된 PhaseVecotr로 설정
        current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
        libsalt.trafficsignal.setTLSPhaseVector(current_step, tlid, current_schedule_id, mpv)

        phase_sum = np.sum([x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector])
        # phase_sum_list.append(phase_sum)
        tl_phase_list = [x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector if
                         x[0] > 5]

        print(tl_phase_list)  ## [29, 72, 17, 48]   …. Green idx가 가리키는 phase의 duration 과 같을 것임…

        tl_phase_list_include_y = [x[0] for x in
                               libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector]
        print(tl_phase_list_include_y)  # [29, 4, 72, 3, 17, 3, 48, 4]  …. 모든 phase의 duration 과 같을 것임…

        phase_arr = []

        for i in range(len(tl_phase_list_include_y)):
            phase_arr = np.append(phase_arr, np.ones(tl_phase_list_include_y[i]) * i)
        print(phase_arr)

        ## offset 만큼 이동
        offset = 4
        new_phase_arr = np.roll(phase_arr, offset)
        print(new_phase_arr)

    if 0:
        duration_info = [69, 4, 4, 59, 3, 3, 14, 3, 3, 38, 4, 4]
        logic_idx = 0
        changeTLPhase(tlid, logic_idx, duration_info)
