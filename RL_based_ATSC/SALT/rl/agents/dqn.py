import sys
sys.path.append('/home/mgpi/traffic-simulator/tools/libsalt')
import libsalt

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import Dropout
from keras.models import Model

from collections import deque
import random
from keras.optimizers import Adam
from keras.optimizers import RMSprop
# from scipy.stats import kurtosis
#import h5py
#import traci
#import math
from keras.layers.normalization import BatchNormalization
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class Learner:
    def __init__(self, action_space_size, exploration, state_d='1d'):
        self.action_size = action_space_size
        self.learning_rate = 0.0001
        if state_d == '1d':
            self.regressor = self._build_model_1d()
            self.regressor_target = self._build_model_1d()
            self.regressor_target.set_weights(self.regressor.get_weights())
        elif state_d == '2d':
            self.regressor = self._build_model_2d()
            self.regressor_target = self._build_model_2d()
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
        # input_1 = Input(shape=(16, 1))
        # x1 = Flatten()(input_1)
        # input_2 = Input(shape=(16, 1))
        # x2 = Flatten()(input_2)
        # input_3 = Input(shape=(16, 1))
        # x3 = Flatten()(input_3)
        # input_4 = Input(shape=(8, 1))
        # x4 = Flatten()(input_4)
        input_1 = Input(shape=(9, 1))
        x1 = Flatten()(input_1)
        input_2 = Input(shape=(9, 1))
        x2 = Flatten()(input_2)
        input_3 = Input(shape=(3, 1))
        x3 = Flatten()(input_3)
        # input_4 = Input(shape=(8, 1))
        # x4 = Flatten()(input_4)
        #input_5 = Input(shape=(8, 1))
        #x5 = Flatten()(input_5)

        x = keras.layers.concatenate([x1, x2, x3])

        x = Dense(100*2, activation='relu')(x)
        x = Dense(80*2, activation='relu')(x)
        x = Dense(60*2, activation='relu')(x)
        x = Dense(40*2, activation='relu')(x)
        x = Dense(20*2, activation='relu')(x)
        #x = Dense(16, activation='relu')(x)
        #x = Dense(16, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)

        regressor = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        regressor.compile(optimizer=keras.optimizers.Adam(
            lr=self.learning_rate), loss='mse')

        return regressor


    def _build_model_2d(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(12, 12, 1))
        x1 = Conv2D(512, (4, 4), strides=(2, 2), activation='relu')(input_1)
        x1 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu')(x1)
        x1 = Flatten()(x1)

        input_2 = Input(shape=(12, 12, 1))
        x2 = Conv2D(512, (4, 4), strides=(2, 2), activation='relu')(input_2)
        x2 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu')(x2)
        x2 = Flatten()(x2)

        input_3 = Input(shape=(12, 12, 1))
        x3 = Conv2D(512, (4, 4), strides=(2, 2), activation='relu')(input_3)
        x3 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu')(x3)
        x3 = Flatten()(x3)

        input_4 = Input(shape=(8, 1))
        x4 = Flatten()(input_4)

        #x = keras.layers.concatenate([x1, x2, x3, x4])
        x = keras.layers.concatenate([x1, x2, x4])
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)

        #regressor = Model(inputs=[input_1, input_2, input_3, input_4], outputs=[x])
        regressor = Model(inputs=[input_1, input_2, input_4], outputs=[x])
        regressor.compile(optimizer=keras.optimizers.RMSprop(
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
        self.regressor.load_weights("model/" + name + ".h5")
        #self.regressor_target.load_weights("model/" + name + "_target.h5")

    def save(self, name):
        self.regressor.save_weights("model/" + name + ".h5")
        self.regressor_target.save_weights("model/" + name + "_target.h5")



def get_state_1d(targetTLNode):
    # vnMatrix = []
    # msMatrix = []
    # wnMatrix = []
    # wqMatrix = []
    # wtMatrix = []
    densityMatrix = []
    passedMatrix = []
    speedMatrix = []
    lightMatrix = []

    link_list = libsalt.trafficsignal.getTLSConnectedLinkID(targetTLNode)

    lane_list = []
    # for link in link_list:
    #     lanes = libsalt.link.getNumLane(link)
    #     for lane in range(lanes):
    #         lane_list = np.append(lane_list, "{}_{}".format(link, lane))

    tl = libsalt.trafficsignal.getTLSByNodeID(targetTLNode)
    tsm = tl.getScheduleMap()
    for k in tsm.keys():
        tl_phase_len = len(tsm[k].getPhaseVector())
    light = np.zeros(tl_phase_len)
    light[libsalt.trafficsignal.getCurrentTLSPhaseIndexByNodeID(targetTLNode)] = 1
    lightMatrix = np.append(lightMatrix, light)

    # for lane in lane_list:
        # vnMatrix = np.append(vnMatrix, libsalt.lane.getAverageNumVehs(lane))
        # msMatrix = np.append(msMatrix, libsalt.lane.getAverageSpeed(lane))
        # wqMatrix = np.append(wqMatrix, libsalt.lane.getAverageWaitingQLength(lane))
        # wtMatrix = np.append(wtMatrix, libsalt.lane.getAverageWaitingTime(lane))
        # msMatrix = np.append(msMatrix, libsalt.network.getAverageSpeed(link))
        # wqMatrix = np.append(wqMatrix, libsalt.link.getAverageWaitingQLength(link))
        # wtMatrix = np.append(wtMatrix, libsalt.link.getAverageWaitingTime(link))
    #print(wqMatrix)
    # currentStep = libsalt.getCurrentStep()
    # lastswitching = libsalt.trafficsignal.getLastTLSPhaseSwitchingTimeByNodeID(targetTLNode)

    #for link in link_list:
    #    wnMatrix = np.append(wnMatrix, libsalt.link.getNumWaitingVehicle(link, currentStep, lastswitching))
    #    wtMatrix = np.append(wtMatrix, libsalt.link.getAverageVehicleWaitingTime(link, currentStep, lastswitching))
    #    wqMatrix = np.append(wqMatrix, libsalt.link.getAverageVehicleWaitingQLength(link, currentStep, lastswitching))

    for link in link_list:
        # wq_link = np.append(wq_link, libsalt.link.getAverageWaitingQLength(link))
        # wt_link = np.append(wt_link, libsalt.link.getAverageWaitingTime(link))
        densityMatrix = np.append(densityMatrix, libsalt.link.getAverageDensity(link))
        passedMatrix = np.append(passedMatrix, libsalt.link.getSumPassed(link))
        speedMatrix = np.append(speedMatrix, libsalt.link.getAverageVehicleSpeed(link))
        # lanes = libsalt.link.getNumLane(link)
        # for lane in range(lanes):
        #    lane_list = np.append(lane_list, "{}_{}".format(link, lane))
        # for s in range(libsalt.link.getNumSection(link)):
        #     for l in range(libsalt.link.getNumLane(link)):
        #         wnMatrix = np.append(wnMatrix,
        #                             libsalt.cell.genNumWaitingVehicle(link, s, l, currentStep, lastswitching))
        #         wtMatrix = np.append(wtMatrix, libsalt.cell.getAverageVehicleWaitingTime(link, s, l, currentStep,
        #                                                                                   lastswitching))
        #         wqMatrix = np.append(wqMatrix, libsalt.cell.getAverageVehicleWaitingQLength(link, s, l, currentStep,
        #                                                                                lastswitching))
    # vehicle_number = np.array(vnMatrix)
    # vehicle_number = vehicle_number.reshape(1, len(vnMatrix), 1)
    #
    # waiting_number = np.array(wnMatrix)
    # waiting_number = waiting_number.reshape(1, len(wnMatrix), 1)
    #
    # mean_speed = np.array(msMatrix)
    # mean_speed = mean_speed.reshape(1, len(msMatrix), 1)
    #
    # waiting_q_length = np.array(wqMatrix)
    # waiting_q_length = waiting_q_length.reshape(1, len(wqMatrix), 1)
    #
    # waiting_time = np.array(wtMatrix)
    # waiting_time = waiting_time.reshape(1, len(wtMatrix), 1)

    density = np.array(densityMatrix)
    density = density.reshape(1, len(densityMatrix), 1)

    passed = np.array(passedMatrix)
    passed = passed.reshape(1, len(passedMatrix), 1)

    speed = np.array(speedMatrix)
    speed = speed.reshape(1, len(speedMatrix), 1)

    lgts = np.array(lightMatrix)
    lgts = lgts.reshape(1, len(lightMatrix), 1)

    #return [vehicle_number, mean_speed, waiting_q_length, waiting_time, lgts]
    return [density, passed, speed, lgts]
    #return [waiting_q_length, waiting_time, lgts]


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

def get_state_2d():
    positionMatrix = []
    velocityMatrix = []
    waitingMatrix = []

    cellLength = 7
    offset = 11
    speedLimit = 14

    junctionPosition = traci.junction.getPosition('572700402')[0]
    vehicles_road1 = traci.edge.getLastStepVehicleIDs('-572700453')
    vehicles_road2 = traci.edge.getLastStepVehicleIDs('-572700451')
    vehicles_road3 = traci.edge.getLastStepVehicleIDs('-572700452')
    vehicles_road4 = traci.edge.getLastStepVehicleIDs('572700400')
    for i in range(12):
        positionMatrix.append([])
        velocityMatrix.append([])
        waitingMatrix.append([])
        for j in range(12):
            positionMatrix[i].append(0)
            velocityMatrix[i].append(0)
            waitingMatrix[i].append(0)

    for v in vehicles_road1:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
        if(ind < 12):
            positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
            velocityMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            waitingMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getWaitingTime(v)

    for v in vehicles_road2:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
        if(ind < 12):
            positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
            velocityMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
            waitingMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getWaitingTime(v)

    junctionPosition = traci.junction.getPosition('572700402')[1]
    for v in vehicles_road3:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
        if(ind < 12):
            positionMatrix[6 + 2 -
                           traci.vehicle.getLaneIndex(v)][11 - ind] = 1
            velocityMatrix[6 + 2 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit
            waitingMatrix[6 + 2 - traci.vehicle.getLaneIndex(v)][11 - ind] = traci.vehicle.getWaitingTime(v)

    for v in vehicles_road4:
        ind = int(
            abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
        if(ind < 12):
            positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
            velocityMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit
            waitingMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getWaitingTime(v)

    light = [0,0,0,0,0,0,0,0]
    light[traci.trafficlight.getPhase('572700402')]=1

    position = np.array(positionMatrix)
    position = position.reshape(1, 12, 12, 1)

    velocity = np.array(velocityMatrix)
    velocity = velocity.reshape(1, 12, 12, 1)

    # waiting = np.array(waitingMatrix)
    # waiting = waiting.reshape(1, 12, 12, 1)

    lgts = np.array(light)
    lgts = lgts.reshape(1, 8, 1)

    #return [position, velocity, waiting, lgts]
    return [position, velocity, lgts]
