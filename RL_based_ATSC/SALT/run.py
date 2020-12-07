import argparse
import libsalt
import os
from pathlib import Path

import numpy as np
import keras
from keras.layers import Input, Flatten, Dense
from keras.models import Model

from collections import deque
import random

cwd_path = os.getcwd()
this_file_dir_path = os.path.dirname(os.path.realpath(__file__))
salt_core_root_dir = os.path.join(this_file_dir_path, "..")

default_conf_path = os.path.join(salt_core_root_dir, "conf", "salt.conf.json")
default_scenario_path = os.path.join(salt_core_root_dir, "data", "salt.scenario.json")
default_output_path = os.path.join(salt_core_root_dir, "output")
default_model_path = os.path.join(salt_core_root_dir, "model")

# TL info of Yuseong Middle School 3
min_green = [26, 53]
max_green = [55, 130]
yellow = [3, 3]
cycle = 120
phase_num = range(3)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Dynamic Simulation")
    parser.add_argument('-c', '--conf', nargs='?', default=default_conf_path)
    parser.add_argument('-s', '--scenario', nargs='?', default=default_scenario_path)

    parser.add_argument('-m', '--mode', choices=['train', 'test', 'fixed'], default='train')
    parser.add_argument('-t', '--targetTL', type=str, default='563103625') # Yuseong Middle School 3
    parser.add_argument('-n', '--model-num', type=str, default='0')
    parser.add_argument('-o', '--optimID', type=str, default='optim_01')
    parser.add_argument('-b', '--beginStep', type=int, default=0)
    parser.add_argument('-e', '--endStep', type=int, default=36000)
    parser.add_argument('-ep', '--epochs', type=int, default=10)

    return parser.parse_args()

class Learner:
    def __init__(self, action_space_size, exploration, state_d='1d'):
        self.action_size = action_space_size
        self.learning_rate = 0.0001
        self.regressor = self._build_model_1d()
        self.regressor_target = self._build_model_1d()
        self.regressor_target.set_weights(self.regressor.get_weights())
        self.exploration = exploration
        self.exploration_decay = 0.99
        self.min_exploration = 0.001
        self.memory = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.95
        self.target_update_counter = 0
        self.target_update_freq = 10

    def _build_model_1d(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(9, 1))
        x1 = Flatten()(input_1)
        input_2 = Input(shape=(9, 1))
        x2 = Flatten()(input_2)
        input_3 = Input(shape=(3, 1))
        x3 = Flatten()(input_3)
        x = keras.layers.concatenate([x1, x2, x3])

        x = Dense(100*2, activation='relu')(x)
        x = Dense(80*2, activation='relu')(x)
        x = Dense(60*2, activation='relu')(x)
        x = Dense(40*2, activation='relu')(x)
        x = Dense(20*2, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)

        regressor = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        regressor.compile(optimizer=keras.optimizers.Adam(
            lr=self.learning_rate), loss='mse')

        return regressor

    def act(self, state):
        if np.random.rand() <= self.exploration:
            action = np.random.choice(range(self.action_size))
        else:
            #print(self.regressor.predict(state))
            action = np.argmax(self.regressor.predict(state), axis=1)[0]
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = random.sample(list(self.memory), self.batch_size)
        # if self.batch_size < 200:
        #    self.batch_size += 10
        for state, action, reward, next_state, done in minibatch:
            # print(action)
            if not done:
                #target = reward + self.gamma * np.max(self.regressor.predict(next_state)[0])
                target = reward + self.gamma*np.max(self.regressor_target.predict(next_state)[0])
            else:
                target = reward
            #target_f = self.regressor.predict(state)
            target_f = self.regressor_target.predict(state)
            target_f[0][action] = target
            self.regressor.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration > self.min_exploration:
            self.exploration *= self.exploration_decay

    def increase_target_update_counter(self):
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.regressor_target.set_weights(self.regressor.get_weights())
            self.target_update_counter = 0

    def load(self, name):
        print(name + ".h5")
        self.regressor.load_weights(name + ".h5")
        #self.regressor_target.load_weights("model/" + name + "_target.h5")

    def save(self, name):
        self.regressor.save_weights(name + ".h5")
        self.regressor_target.save_weights(name + "_target.h5")


def get_state_1d_v2(targetTLNode):
    inWaitingNumberMatrix = []
    outWaitingNumberMatrix = []
    lightMatrix = []

    in_link_list = ['-563104367', '-563104373', '-563105308']
    out_link_list = ['563104367', '563104373', '563105308']

    tl = libsalt.trafficsignal.getTLSByNodeID(targetTLNode)
    tsm = tl.getScheduleMap()
    for k in tsm.keys():
        tl_phase_len = len(tsm[k].getPhaseVector())
    light = np.zeros(tl_phase_len)
    light[libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)] = 1
    lightMatrix = np.append(lightMatrix, light)

    for inl in in_link_list:
        numLane = libsalt.link.getNumLane(inl)
        numSection = libsalt.link.getNumSection(inl)
        for nl in range(numLane):
            for ns in range(numSection):
                cell_id = inl + '_' + str(ns) + '_' + str(nl)
                inWaitingNumberMatrix = np.append(inWaitingNumberMatrix, libsalt.cell.getAverageNumVehicles(cell_id))

    for onl in out_link_list:
        numLane = libsalt.link.getNumLane(onl)
        numSection = libsalt.link.getNumSection(onl)
        for nl in range(numLane):
            for ns in range(numSection):
                cell_id = onl + '_' + str(ns) + '_' + str(nl)
                outWaitingNumberMatrix = np.append(outWaitingNumberMatrix, libsalt.cell.getAverageNumVehicles(cell_id))

    incoming = np.array(inWaitingNumberMatrix)
    incoming = incoming.reshape(1, len(inWaitingNumberMatrix), 1)

    outgoing = np.array(outWaitingNumberMatrix)
    outgoing = outgoing.reshape(1, len(outWaitingNumberMatrix), 1)

    lgts = np.array(lightMatrix)
    lgts = lgts.reshape(1, len(lightMatrix), 1)

    return [incoming, outgoing, lgts]


def getReward(in_link_list, out_link_list):
    sum_dict = {}
    for inl in in_link_list:
        numLane = libsalt.link.getNumLane(inl)
        numSection = libsalt.link.getNumSection(inl)
        sum_dict[inl] = 0
        for nl in range(numLane):
            for ns in range(numSection):
                cell_id = inl + '_' + str(ns) + '_' + str(nl)
                sum_dict[inl] += libsalt.cell.getAverageNumVehicles(cell_id)
    for onl in out_link_list:
        numLane = libsalt.link.getNumLane(onl)
        numSection = libsalt.link.getNumSection(onl)
        sum_dict[onl] = 0
        for nl in range(numLane):
            for ns in range(numSection):
                cell_id = onl + '_' + str(ns) + '_' + str(nl)
                sum_dict[onl] += libsalt.cell.getAverageNumVehicles(cell_id)
    ns = sum_dict['-563104367']# - sum_dict['563105308']
    sn = sum_dict['-563105308']# - sum_dict['563104367']
    e = sum_dict['-563104373']

    reward = -np.abs(ns+sn+e)
    return np.round(reward, 2)

def main():
    args = parse_args()
    confPath = os.path.join(cwd_path, args.conf)
    scenarioPath = os.path.join(cwd_path, args.scenario)
    output_path = os.path.join(default_output_path, args.optimID)

    beginStep = args.beginStep
    endStep = args.endStep
    epochs = args.epochs

    targetTLNode = args.targetTL
    optimID = args.optimID

    output_filename_reward = os.path.join(default_output_path, args.optimID, "output-reward.csv")
    output_filename_phase_rl = os.path.join(default_output_path, args.optimID, "output-phase-rl.csv")
    output_filename_phase_ft = os.path.join(default_output_path, args.optimID, "output-phase-ft.csv")
    output_dir = s.path.join(default_output_path, args.optimID)
    model_dir = os.path.join(default_model_path, args.optimID)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # print('args', args)
    # print('conf path', confPath)
    # print('scenario path', scenarioPath)

    # ======================================

    action_space_size = 15  # (0-14)
    agent = Learner(action_space_size, 1, '1d')

    in_link_list = ['-563104367', '-563104373', '-563105308']
    out_link_list = ['563104367', '563104373', '563105308']

    if args.mode == 'train':
        with open(output_filename_reward, 'w+') as f:
            f.write("Simulation,reward_sum\n")
        for simulation in range(epochs):
            libsalt.start(scenarioPath)
            libsalt.setCurrentStep(beginStep)
            step = libsalt.getCurrentStep()
            total_reward = 0
            link_list = libsalt.trafficsignal.getTLSConnectedLinkID(targetTLNode)
            current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)
            current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(targetTLNode)

            print("cycle init start")
            for i in phase_num:
                while True:
                    current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)
                    if current_phase != i:
                        break
                    current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(targetTLNode)
                    libsalt.simulationStep()
                    step = libsalt.getCurrentStep()
                    # print("step {} current_phase {}".format(simulationSteps, current_phase))
            print("cycle init end")

            done = 0
            cycle_idx = 0
            while step <= endStep:
                state = get_state_1d_v2(targetTLNode)
                action = agent.act(state)

                p1_duration = min_green[0] + action * 2  # (26-54)
                p2_duration = cycle - p1_duration  # (94-66)

                phase_arr = [p1_duration, p2_duration]

                reward_list = []
                for idx, duration in enumerate(phase_arr):
                    for i in range(duration):
                        libsalt.trafficsignal.changeTLSPhase(step, targetTLNode, current_schedule_id, idx)
                        libsalt.simulationStep()
                        step = libsalt.getCurrentStep()
                        reward_tmp = getReward(in_link_list, out_link_list)
                        reward_list = np.append(reward_list, reward_tmp)

                reward = np.mean(reward_list)  # + p2_duration*0.1
                reward = np.round(reward, 2)
                # print(reward)
                total_reward += reward

                next_state = get_state_1d_v2(targetTLNode)

                agent.remember(state, action, reward, next_state, done)

                if cycle_idx % 8 == 0 and agent.batch_size < len(agent.memory):
                    agent.increase_target_update_counter()
                    agent.replay()
                cycle_idx += 1

            libsalt.close()

            with open(output_filename_reward, "a") as f:
                f.write("{},{}\n".format(simulation, np.round(total_reward, 2)))
                model_path = os.path.join(model_dir, "traffic_epoch{}".format(simulation))
                agent.save(model_path)

            print("Python: Simulation End!!!")
        print("Python: Training End!!!")

    elif args.mode == 'test':
        agent = Learner(action_space_size, 0, '1d')

        libsalt.start(scenarioPath)
        libsalt.setCurrentStep(beginStep)
        step = libsalt.getCurrentStep()
        total_reward = 0
        link_list = libsalt.trafficsignal.getTLSConnectedLinkID(targetTLNode)
        current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)
        current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(targetTLNode)

        print("cycle init start")
        for i in phase_num:
            while True:
                current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)
                if current_phase != i:
                    break
                current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(targetTLNode)
                libsalt.simulationStep()
                step = libsalt.getCurrentStep()
                # print("step {} current_phase {}".format(simulationSteps, current_phase))
        print("cycle init end")

        with open(output_filename_phase_rl, "w+") as f:
            f.write("SimulationStep,action,p1_duration,p2_duration\n")

        model_num = args.model_num
        trained_model_path = os.path.join(model_dir, "traffic_epoch{}".format(model_num))
        print(trained_model_path)
        agent.load(trained_model_path)

        done = 0
        cycle_idx = 0
        while step <= endStep:
            state = get_state_1d_v2(targetTLNode)
            action = agent.act(state)

            p1_duration = min_green[0] + action * 2  # (26-54)
            p2_duration = cycle - p1_duration  # (94-66)

            phase_arr = [p1_duration, p2_duration]

            with open(output_filename_phase_rl, "a") as f:
                f.write("{},{},{},{}\n".format(step, action, p1_duration, p2_duration))

            reward_list = []
            for idx, duration in enumerate(phase_arr):
                for i in range(duration):
                    libsalt.trafficsignal.changeTLSPhase(step, targetTLNode, current_schedule_id, idx)
                    libsalt.simulationStep()
                    step = libsalt.getCurrentStep()
                    reward_tmp = getReward(in_link_list, out_link_list)
                    reward_list = np.append(reward_list, reward_tmp)

            reward = np.mean(reward_list)  # + p2_duration*0.1
            reward = np.round(reward, 2)
            # print(reward)
            total_reward += reward

        libsalt.close()
        print("Python: Simulation End!!!")
    # ======================================

    elif args.mode == 'fixed':
        libsalt.start(scenarioPath)
        libsalt.setCurrentStep(beginStep)
        step = libsalt.getCurrentStep()
        link_list = libsalt.trafficsignal.getTLSConnectedLinkID(targetTLNode)
        current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)

        print("cycle init start")
        for i in phase_num:
            while True:
                current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)
                if current_phase != i:
                    break
                current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(targetTLNode)
                libsalt.simulationStep()
                step = libsalt.getCurrentStep()
                #print("step {} current_phase {}".format(step, current_phase))
        print("cycle init end")

        with open(output_filename_phase_ft, "w+") as f:
            f.write("SimulationStep,p1_duration,p2_duration\n")

        while step <= endStep:
            p1_duration = 35
            p2_duration = 85

            phase_arr = [p1_duration, p2_duration]

            with open(output_filename_phase_ft, "a") as f:
                f.write("{},{},{}\n".format(step, p1_duration, p2_duration))

            for idx, duration in enumerate(phase_arr):
                for i in range(duration):
                    current_schedule_id = libsalt.trafficsignal.getCurrentTLSScheduleIDByNodeID(targetTLNode)
                    # libsalt.trafficsignal.changeTLSPhase(simulationSteps, targetTLNode, current_schedule_id, 1)
                    current_phase = libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)
                    print("step {} current_phase {}".format(step, current_phase))
                    libsalt.simulationStep()
                    step = libsalt.getCurrentStep()

        libsalt.close()

    exit(0)

main()

