# -*- coding: utf-8 -*-

import numpy as np

import libsalt

from deprecated import deprecated

from DebugConfiguration import DBG_OPTIONS


class SaltActionMgmt:
    '''
    a class for Salt action mgmt
    '''
    def __init__(self, args, sa_obj, sa_name_list):
        '''
        constructor
        :param args:
        :param sa_obj:
        :param sa_name_list:
        '''
        self.args = args
        self.sa_obj = sa_obj
        self.sa_name_list = sa_name_list

        #-- to save phase info for given fixed signal : [0,0, ,,,0, 1, 1, 1, 2, 2, 2,,,,,2, 3, 3, 3, 4,...]
        self.initial_phase_array_list = []
        #-- to save phase info which are action applied
        self.apply_phase_array_list = []  # env.sa_phase_arr_list

        for sa in self.sa_name_list:
            # -- initial phase array of SAs : phase info of all TLs within SAs
            sa_phase_array = self.__getInitialPhaseArray(self.sa_obj[sa])
            self.initial_phase_array_list.append(sa_phase_array)
            offset_list = self.sa_obj[sa]['offset_list']
            apply_phase_array = self.__getOffsetAppliedPhaseArray(sa_phase_array, offset_list)
            self.apply_phase_array_list.append(apply_phase_array)


    def __getInitialPhaseArray(self, an_sa_obj):
        '''
        get a phase array list of given SA
        :param an_sa_obj:
        :return:
        '''
        phase_arr_list = []
        tlid_list = an_sa_obj['tlid_list']

        # remove unnecessary variable : tlid
        for tlid_i in range(len(tlid_list)):
            currDur = an_sa_obj['duration_list'][tlid_i]

            phase_arr = []
            for i in range(len(currDur)):
                phase_arr = np.append(phase_arr, np.ones(currDur[i]) * i)

            phase_arr_list.append(phase_arr)

        return phase_arr_list



    def __getOffsetAppliedPhaseArray(self, in_phase_arr_list, offset_list):
        '''
        convert a give phase array list into a offset applied phase array list
        :param in_phase_arr_list:
        :param offset_list:
        :return:
        '''
        out_phase_arr_list = []

        for i in range(len(offset_list)):
            out_phase_arr = np.roll(in_phase_arr_list[i], offset_list[i])
            out_phase_arr_list.append(out_phase_arr)

        return out_phase_arr_list


    def __getGreenRatioAppliedPhaseArray(self, curr_sim_step, an_sa_obj, actions):
        '''
        get green-ratio actions applied phase array list

        :param curr_sim_step: current sumulation step
        :param an_sa_obj: object which holds information about an SA
        :param actions: actions to apply
        :return:
        '''
        #print('actions')
        #print(actions)
        tlid_list = an_sa_obj["tlid_list"]
        # sa_cycle = an_sa_obj["cycle_list"][0]
        #print('tlid_list')
        #print(tlid_list)
        
        phase_sum_list = []
        phase_list = []
        phase_array_list = []

        if DBG_OPTIONS.RichActionOutput:
            duration_list=[]
        #print('tlid', 'green_idx', 'min_dur', 'max_dur', 'curDur')
        for tlid_idx in range(len(tlid_list)):
            tlid = tlid_list[tlid_idx]
            green_idx = an_sa_obj["green_idx_list"][tlid_idx][0]
            minDur = an_sa_obj["minDur_list"][tlid_idx]
            maxDur = an_sa_obj['maxDur_list'][tlid_idx]
            currDur = an_sa_obj['duration_list'][tlid_idx]
            #print(tlid, green_idx, minDur, maxDur, currDur)
            
            if DBG_OPTIONS.RichActionOutput:
                new_duration = currDur.copy()

            mpv = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector
            mpv = list(mpv)

            action_list = an_sa_obj['action_list_list'][tlid_idx]
            action = action_list[actions[tlid_idx]]

            for _i in range(len(green_idx)):
                gi = green_idx[_i]
                _m = list(mpv[gi])
                _m[0] = currDur[gi] + int(action[_i]) * self.args.add_time
                mpv[gi] = tuple(_m)

                if DBG_OPTIONS.RichActionOutput:
                    new_duration[gi]=_m[0]
            if DBG_OPTIONS.RichActionOutput:
                duration_list.append(new_duration)
            #print('mpv'); print(mpv)
            scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
            #libsalt.trafficsignal.setTLSPhaseVector(curr_sim_step, tlid, scheduleID, mpv)
            libsalt.trafficsignal.changeTLSPhaseVectorInStepwise(curr_sim_step, tlid, scheduleID, mpv)

            phase_sum = np.sum([x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector])
            phase_sum_list.append(phase_sum)
            tl_phase_list = [x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector if
                             x[0] > 5]
                    # todo : should avoid CONSTANT 5... it reduce readibility
            phase_list.append(tl_phase_list)
            tl_phase_list_include_y = [x[0] for x in
                                       libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector]
            #print(tl_phase_list_include_y)
            phase_arr = []

            for i in range(len(tl_phase_list_include_y)):
                phase_arr = np.append(phase_arr, np.ones(tl_phase_list_include_y[i]) * i)
            phase_array_list.append(np.roll(phase_arr, an_sa_obj['offset_list'][tlid_idx]))

        if DBG_OPTIONS.RichActionOutput:
            return phase_array_list, duration_list
        else:
            return phase_array_list


    def __setGreenTimePhaseArray(self, curr_sim_step, an_sa_obj, actions):
        '''
        set green-time actions applied phase array list

        :param curr_sim_step: current sumulation step
        :param an_sa_obj: object which holds information about an SA
        :param actions: actions to apply
        :return:
        '''
        #print('actions')
        #print(actions)
        tlid_list = an_sa_obj["tlid_list"]
        # sa_cycle = an_sa_obj["cycle_list"][0]
        #print('tlid_list')
        #print(tlid_list)
        
        #phase_sum_list = []
        #phase_list = []
        phase_array_list = []

        action_list = []
        start = 0
        #rearange actions for each intersection.
        for size in an_sa_obj["action_space_list"]:
            #print('size', size)
            sub_action = actions[start:start+size] 
            start += size
            action_list.append(sub_action)
            
        #print('action_list')
        #print(action_list)
        
        if DBG_OPTIONS.RichActionOutput:
            duration_list=[]
        
        #print('tlid', 'green_idx', 'min_dur', 'max_dur', 'curDur')
        #for tlid_idx in range(len(tlid_list)):
        for tlid_idx, (tlid, action) in enumerate(zip(tlid_list, action_list)):
            tlid = tlid_list[tlid_idx]
            green_idx = an_sa_obj["green_idx_list"][tlid_idx][0]
            minDur = an_sa_obj["minDur_list"][tlid_idx]
            maxDur = an_sa_obj['maxDur_list'][tlid_idx]
            currDur = an_sa_obj['duration_list'][tlid_idx]
            #print(tlid, green_idx, minDur, maxDur, currDur)
            
            if DBG_OPTIONS.RichActionOutput:
                new_duration = currDur.copy()

            mpv = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector
            mpv = list(mpv)

            #action_list = an_sa_obj['action_list_list'][tlid_idx]
            #action = action_list[actions[tlid_idx]]
            #print('action', action)
            for _i in range(len(green_idx)):
                gi = green_idx[_i]
                _m = list(mpv[gi])
                median = 0.5 * (maxDur[gi] + minDur[gi]); time_span = maxDur[gi] - median # = median - minDur[gi]
                #print('action[_i]', action[_i])
                _m[0] = int(median + time_span * action[_i])
                mpv[gi] = tuple(_m)

                if DBG_OPTIONS.RichActionOutput:
                    new_duration[gi]=_m[0]
            if DBG_OPTIONS.RichActionOutput:
                duration_list.append(new_duration)
            #print('mpv'); print(mpv)
            
            scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
            #libsalt.trafficsignal.setTLSPhaseVector(curr_sim_step, tlid, scheduleID, mpv)
            libsalt.trafficsignal.changeTLSPhaseVectorInStepwise(curr_sim_step, tlid, scheduleID, mpv)

            #phase_sum = np.sum([x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector])
            #phase_sum_list.append(phase_sum)
            #tl_phase_list = [x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector if
            #                 x[0] > 5]
                    # todo : should avoid CONSTANT 5... it reduce readibility
            #phase_list.append(tl_phase_list)
            
            tl_phase_list_include_y = [x[0] for x in
                                       libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector]
            #print(tl_phase_list_include_y)
            phase_arr = []
            for i in range(len(tl_phase_list_include_y)):
                phase_arr = np.append(phase_arr, np.ones(tl_phase_list_include_y[i]) * i)
            phase_array_list.append(np.roll(phase_arr, an_sa_obj['offset_list'][tlid_idx]))

        if DBG_OPTIONS.RichActionOutput:
            return phase_array_list, duration_list
        else:
            return phase_array_list


    def __setDefaultGreenTimePhaseArray(self, curr_sim_step, an_sa_obj, actions=None):
        '''
        set green-time actions applied phase array list

        :param curr_sim_step: current sumulation step
        :param an_sa_obj: object which holds information about an SA
        :param actions: actions to apply
        :return:
        '''
        #print('actions')
        #print(actions)
        tlid_list = an_sa_obj["tlid_list"]
        # sa_cycle = an_sa_obj["cycle_list"][0]
        #print('tlid_list')
        #print(tlid_list)
        
        #phase_sum_list = []
        #phase_list = []
        phase_array_list = []

#        action_list = []
#        start = 0
#        #rearange actions for each intersection.
#        for size in an_sa_obj["action_space_list"]:
#            #print('size', size)
#            sub_action = actions[start:start+size] 
#            start += size
#            action_list.append(sub_action)
            
        #print('action_list')
        #print(action_list)
        
        if DBG_OPTIONS.RichActionOutput:
            duration_list=[]
        
        #print('tlid', 'green_idx', 'min_dur', 'max_dur', 'curDur')
        #for tlid_idx in range(len(tlid_list)):
        #for tlid_idx, (tlid, action) in enumerate(zip(tlid_list, action_list)):
        for tlid_idx, tlid in enumerate(tlid_list):
            tlid = tlid_list[tlid_idx]
            green_idx = an_sa_obj["green_idx_list"][tlid_idx][0]
            minDur = an_sa_obj["minDur_list"][tlid_idx]
            maxDur = an_sa_obj['maxDur_list'][tlid_idx]
            currDur = an_sa_obj['duration_list'][tlid_idx]
            #print(tlid, green_idx, minDur, maxDur, currDur)
            
            if DBG_OPTIONS.RichActionOutput:
                new_duration = currDur.copy()

            mpv = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector
            mpv = list(mpv)

            #action_list = an_sa_obj['action_list_list'][tlid_idx]
            #action = action_list[actions[tlid_idx]]
            #print('action', action)
            for _i in range(len(green_idx)):
                gi = green_idx[_i]
                _m = list(mpv[gi])
                #median = 0.5 * (maxDur[gi] + minDur[gi]); time_span = maxDur[gi] - median # = median - minDur[gi]
                #print('action[_i]', action[_i])
                #_m[0] = int(median + time_span * action[_i])
                _m[0] = currDur[gi]
                mpv[gi] = tuple(_m)

                if DBG_OPTIONS.RichActionOutput:
                    new_duration[gi]=_m[0]
            if DBG_OPTIONS.RichActionOutput:
                duration_list.append(new_duration)
            #print('mpv'); print(mpv)
            
            scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
            #libsalt.trafficsignal.setTLSPhaseVector(curr_sim_step, tlid, scheduleID, mpv)
            libsalt.trafficsignal.changeTLSPhaseVectorInStepwise(curr_sim_step, tlid, scheduleID, mpv)

            #phase_sum = np.sum([x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector])
            #phase_sum_list.append(phase_sum)
            #tl_phase_list = [x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector if
            #                 x[0] > 5]
                    # todo : should avoid CONSTANT 5... it reduce readibility
            #phase_list.append(tl_phase_list)
            
            tl_phase_list_include_y = [x[0] for x in
                                       libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector]
            #print(tl_phase_list_include_y)
            phase_arr = []
            for i in range(len(tl_phase_list_include_y)):
                phase_arr = np.append(phase_arr, np.ones(tl_phase_list_include_y[i]) * i)
            phase_array_list.append(np.roll(phase_arr, an_sa_obj['offset_list'][tlid_idx]))

        if DBG_OPTIONS.RichActionOutput:
            return phase_array_list, duration_list
        else:
            return phase_array_list


    @deprecated
    def __getGreenRatioOffsetAppliedPhaseArrayV1(self, curr_sim_step, an_sa_obj, actions):
        '''
        get green-ratio and offset actions applied phase array list

        :param curr_sim_step:
        :param an_sa_obj:
        :param actions:
        :return:
        '''
        ## actions = [offset_0, green_ratio_0, offset_1, green_ratio_1, .., ]
        tlid_list = an_sa_obj["tlid_list"]

        phase_sum_list = []
        phase_list = []
        phase_array_list = []
        for tlid_idx in range(len(tlid_list)):
            tlid = tlid_list[tlid_idx]
            green_idx = an_sa_obj["green_idx_list"][tlid_idx][0]
            # min_dur = an_sa_obj["minDur_list"][tlid_idx]
            # maxDur = an_sa_obj['maxDur_list'][tlid_idx]
            currDur = an_sa_obj['duration_list'][tlid_idx]

            mpv = libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector
            mpv = list(mpv)

            action_list = an_sa_obj['action_list_list'][tlid_idx]
            action = action_list[actions[tlid_idx*2+1]]      ## get an action to be used for adjustment of green ratio

            for _i in range(len(green_idx)):
                gi = green_idx[_i]
                _m = list(mpv[gi])
                _m[0] = currDur[gi] + int(action[_i]) * self.args.add_time
                mpv[gi] = tuple(_m)

            scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
            libsalt.trafficsignal.setTLSPhaseVector(curr_sim_step, tlid, scheduleID, mpv)

            phase_sum = np.sum([x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector])
            phase_sum_list.append(phase_sum)
            tl_phase_list = [x[0] for x in libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector if
                             x[0] > 5]
            phase_list.append(tl_phase_list)
            tl_phase_list_include_y = [x[0] for x in
                                       libsalt.trafficsignal.getCurrentTLSScheduleByNodeID(tlid).myPhaseVector]
            phase_arr = []

            for i in range(len(tl_phase_list_include_y)):
                phase_arr = np.append(phase_arr, np.ones(tl_phase_list_include_y[i]) * i)
            offset_shift = actions[tlid_idx*2]  #  ## get an action to be used for adjustment of offset
            phase_array_list.append(np.roll(phase_arr, an_sa_obj['offset_list'][tlid_idx] + offset_shift ))  ##

        return phase_array_list



    def __getGreenRatioOffsetAppliedPhaseArray(self, curr_sim_step, an_sa_obj, actions):
        '''
        get green-ratio and offset actions applied phase array list
        (code reuse version) it uses __getGreenRatioAppliedPhaseArray() and __getOffsetAppliedPhaseArray()
        :param curr_sim_step:
        :param an_sa_obj:
        :param actions:
        :return:
        '''

        ## reuse code : gro = gr + offset
        ## actions = [offset_0, green_ratio_0, offset_1, green_ratio_1, .., ]

        ## 1. separate actions into two
        #     : actions for green ratio adjustment, actions for offset adjustment
        actions_len = len(actions)
        gr_actions = []
        offset_acctions = []
        for i in range(actions_len):
            if i % 2:
                gr_actions.append(actions[i])
            else:
                offset_acctions.append(actions[i])

        ## 2. do green ratio adjustment
        if DBG_OPTIONS.RichActionOutput:
            in_phase_array_list,  duration_list = self.__getGreenRatioAppliedPhaseArray(curr_sim_step, an_sa_obj, gr_actions)
        else:
            in_phase_array_list = self.__getGreenRatioAppliedPhaseArray(curr_sim_step, an_sa_obj, gr_actions)

        ## 3. do offset adjustment
        # offset + action
        offset_list = [x + y for x, y in zip(an_sa_obj['offset_list'], offset_acctions)]

        if DBG_OPTIONS.PrintAction:
            print(f"DBG offset_list={an_sa_obj['offset_list']} orignal")

        if 0: # changed 20220720PM : todo should check which one is correct
            phase_array_list = self.__getOffsetAppliedPhaseArray(in_phase_array_list, offset_acctions)
        else:
            phase_array_list = self.__getOffsetAppliedPhaseArray(in_phase_array_list, offset_list)

        if DBG_OPTIONS.RichActionOutput:
            return phase_array_list, offset_list, duration_list
        else:
            return phase_array_list



    def applyCurrentTrafficSignalPhaseToEnv(self, current_sim_step):
        '''
        apply actions for all TLs : offset, gr, gro

        :param current_sim_step:
        :return:
        '''
        return 0
        num_sa = len(self.sa_name_list)

        for sa_i in range(num_sa):
            sa = self.sa_name_list[sa_i]
            tlid_list = self.sa_obj[sa]['tlid_list']
            #print('tlid_list'); print(tlid_list)
            tlid_i = 0
            sa_cycle = self.sa_obj[sa]['cycle_list'][0]
            phase_arr = self.apply_phase_array_list[sa_i]

            for tlid in tlid_list:
                #t_phase = int(phase_arr[tlid_i][current_sim_step % sa_cycle])
                t_phase = int(phase_arr[tlid_i][(current_sim_step-1) % sa_cycle])
                scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
                libsalt.trafficsignal.changeTLSPhase(current_sim_step, tlid, scheduleID, t_phase)
                tlid_i += 1

        return 0


    def applyKeepChangeActionFirstStep(self, current_sim_step, all_actions, target_tl_obj):
        '''
        apply keep-change actions : first step

        :param current_sim_step:
        :param all_actions:
        :param target_tl_obj:
        :return: current phase index : list of list
        '''
        phase_list = []
        num_sa = len(self.sa_name_list)
        for sa_idx in range(num_sa):
            sa = self.sa_name_list[sa_idx]
            actions = all_actions[sa_idx]
            tlid_list = self.sa_obj[sa]['tlid_list']
            tmp_phase_list = []
            for tlid_idx in range(len(tlid_list)):
                tlid = tlid_list[tlid_idx]
                scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
                phase_length = len(target_tl_obj[tlid]['duration'])
                current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)
                next_phase = (current_phase + actions[tlid_idx]) % phase_length
                libsalt.trafficsignal.changeTLSPhase(current_sim_step, tlid, scheduleID, int(next_phase))
                tmp_phase_list = np.append(tmp_phase_list, current_phase)  # store current phase
            phase_list.append(tmp_phase_list)
        return phase_list


    def applyKeepChangeActionSecondStep(self, current_sim_step, all_actions, tl_obj):
        '''
        apply keep-change actions : second step

        :param current_sim_step:
        :param all_actions:
        :param tl_obj:
        :return: next phase index : list of list
        '''
        phase_list = []
        num_sa = len(self.sa_name_list)
        for sa_idx in range(num_sa):
            sa = self.sa_name_list[sa_idx]
            actions = all_actions[sa_idx]
            tlid_list = self.sa_obj[sa]['tlid_list']
            tmp_phase_list = []
            for tlid_idx in range(len(tlid_list)):
                tlid = tlid_list[tlid_idx]
                scheduleID = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(tlid)
                phase_length = len(tl_obj[tlid]['duration'])
                current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(tlid)
                if tl_obj[tlid]['duration'][current_phase] > 5:
                    next_phase = current_phase
                else:
                    next_phase = (current_phase + actions[tlid_idx]) % phase_length
                libsalt.trafficsignal.changeTLSPhase(current_sim_step, tlid, scheduleID, int(next_phase))
                tmp_phase_list = np.append(tmp_phase_list, next_phase)  # store next phase
            phase_list.append(tmp_phase_list)
        return phase_list



    def changePhaseArray(self, curr_sim_step, sa_idx, actions=None):
        '''
        change phase array based on given action

        :param curr_sim_step:
        :param sa_idx:
        :param actions: discrete action
        :return:
        '''
        sa = self.sa_name_list[sa_idx]
        an_sa_obj = self.sa_obj[sa]

        if DBG_OPTIONS.RichActionOutput:
            offset_list = []
            duration_list = []

        if DBG_OPTIONS.PrintAction:
            print(f'DBG actions={actions}')

        if self.args.action == 'offset':
            # offset + action
            if DBG_OPTIONS.PrintAction:
                print(f"DBG offset_list={an_sa_obj['offset_list']} orignal")
            offset_list = [x + y for x, y in zip(an_sa_obj['offset_list'], actions)]
            sa_phase_array = self.initial_phase_array_list[sa_idx]
            self.apply_phase_array_list[sa_idx] = self.__getOffsetAppliedPhaseArray(sa_phase_array, offset_list)

        elif self.args.action == 'kc':  # keep or change(limit phase sequence) : ref. step() at sappo_noConst.py
            pass # nothing to do

        elif self.args.action == 'fx':  # fxied default signal
            if DBG_OPTIONS.RichActionOutput:
                self.apply_phase_array_list[sa_idx], duration_list = self.__setDefaultGreenTimePhaseArray(curr_sim_step, an_sa_obj, actions)
            else:
                self.apply_phase_array_list[sa_idx] = self.__setDefaultGreenTimePhaseArray(curr_sim_step, an_sa_obj, actions)

        elif self.args.action == 'gt':  # 
            if DBG_OPTIONS.RichActionOutput:
                self.apply_phase_array_list[sa_idx], duration_list = self.__setGreenTimePhaseArray(curr_sim_step, an_sa_obj, actions)
            else:
                self.apply_phase_array_list[sa_idx] = self.__setGreenTimePhaseArray(curr_sim_step, an_sa_obj, actions)

        elif self.args.action == 'gr':  # green ratio : ref. step() at sappo_green_single.py
            if DBG_OPTIONS.RichActionOutput:
                self.apply_phase_array_list[sa_idx], duration_list = self.__getGreenRatioAppliedPhaseArray(curr_sim_step, an_sa_obj, actions)
            else:
                self.apply_phase_array_list[sa_idx] = self.__getGreenRatioAppliedPhaseArray(curr_sim_step, an_sa_obj,
                                                                                            actions)
        elif self.args.action == 'gro':  # green ratio+offset : ref. step() at sappo_green_offset_single.py
            if DBG_OPTIONS.RichActionOutput:
                self.apply_phase_array_list[sa_idx], offset_list, duration_list = self.__getGreenRatioOffsetAppliedPhaseArray(curr_sim_step, an_sa_obj, actions)
            else:
                self.apply_phase_array_list[sa_idx] = self.__getGreenRatioOffsetAppliedPhaseArray(curr_sim_step,
                                                                                                  an_sa_obj, actions)

        if DBG_OPTIONS.RichActionOutput:
            if DBG_OPTIONS.PrintAction:
                print(f'DBG offset_list_{sa_idx}={offset_list} changed ... in ActionMgmt')
            if len(duration_list):
                if DBG_OPTIONS.PrintAction:
                    print(f'DBG duration_list_{sa_idx}={duration_list} changed  ... in ActionMgmt')

            return offset_list, duration_list
        else:
            return 0



    def convertToDiscreteAction(self, sa_name, actions):
        '''
        convert actions to discrete actions
        :param sa_name:
        :param actions:
        :return:
        '''
        discrete_action = []
        sa_cycle = self.sa_obj[sa_name]['cycle_list'][0]
        sa_action_list_list = self.sa_obj[sa_name]['action_list_list']  # green time adjust table
           #-- 각 교차로의 녹색 시간 조정 action list(주 현시와 나머지 현시 조정)

        for i in range(len(actions)):
            if self.args.action == 'offset':
                # discrete_action.append(int(np.round(actions[i][di] * sa_cycle[i]) / 2 / args.offset_range))
                discrete_action.append(int(np.round(actions[i]*sa_cycle)/2/self.args.offset_range))
            elif self.args.action == 'kc':  # keep or change(limit phase sequence)
                # discrete_action.append(0 if actions[i][di] < args.actionp else 1)
                discrete_action.append(0 if actions[i] < self.args.actionp else 1)
            elif self.args.action == 'gr': # green ratio,
                discrete_action.append(np.digitize(actions[i],
                                        bins=np.linspace(-1, 1, len(sa_action_list_list[i]))) - 1)
            elif self.args.action == 'gt': # green time,
                discrete_action.append(actions[i]) # discretizaton is not required

            elif self.args.action == 'fx': # fixed default signal 
                discrete_action.append(actions[i]) # discretizaton is not required
                
            elif self.args.action == 'gro':  # green ratio+offset
                if ( i % 2 ) == 0:  #  if 0 or even number, it is for offset
                    discrete_action.append(int(np.round(actions[i] * sa_cycle) / 2 / self.args.offset_range))
                else:  # ( i % 2) == 1... if odd number, it is for green ratio
                    discrete_action.append(np.digitize(actions[i],
                                                       bins=np.linspace(-1, 1, len(sa_action_list_list[int(i/2)]))) - 1)
        return discrete_action
