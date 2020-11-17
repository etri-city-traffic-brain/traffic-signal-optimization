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

import h5py
import traci
import math
from keras.layers.normalization import BatchNormalization


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
        self.exploration_decay = 0.95
        self.min_exploration = 0.1
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.95
        self.target_update_counter = 0
        self.target_update_freq = 10

    def _build_model_1d(self):
        input_1 = Input(shape=(58, 1))
        x1 = Flatten()(input_1)
        input_2 = Input(shape=(58, 1))
        x2 = Flatten()(input_2)
        input_3 = Input(shape=(8, 1))
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])

        x = Dense(20*5, activation='relu')(x)
        x = Dense(20*4, activation='relu')(x)
        x = Dense(20*3, activation='relu')(x)
        x = Dense(20*2, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)

        regressor = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        regressor.compile(optimizer=keras.optimizers.Adam(
            lr=self.learning_rate), loss='mse')

        return regressor

    def act(self, state):
        if np.random.rand() <= self.exploration:
            action = np.random.choice(range(self.action_size))
        else:
            action = np.argmax(self.regressor.predict(state), axis=1)[0]
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = random.sample(list(self.memory), self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = reward + self.gamma*np.max(self.regressor_target.predict(next_state)[0])
            else:
                target = reward
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

    def save(self, name):
        self.regressor.save_weights("model/" + name + ".h5")
        self.regressor_target.save_weights("model/" + name + "_target.h5")


def get_state_1d(tl_list):
    vnMatrix = []
    hnMatrix = []
    lightMatrix = []

    for tl in tl_list:
        lane_list = traci.trafficlight.getControlledLanes(tl)
        
        for l in lane_list:
            vnMatrix = np.append(vnMatrix, traci.lane.getLastStepVehicleNumber(l))
            hnMatrix = np.append(hnMatrix, traci.lane.getLastStepHaltingNumber(l))

        tl_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)
        tl_phase_state_len = len(traci.trafficlight.getControlledLanes(tl))
        tl_phase_len = len(tl_logic[0].phases)

        light = np.zeros(tl_phase_len)
        light[traci.trafficlight.getPhase(tl)] = 1
        lightMatrix = np.append(lightMatrix, light)

        vehicle_number = np.array(vnMatrix)
        vehicle_number = vehicle_number.reshape(1, len(vnMatrix), 1)

        halting_number = np.array(hnMatrix)
        halting_number = halting_number.reshape(1, len(hnMatrix), 1)

        lgts = np.array(lightMatrix)
        lgts = lgts.reshape(1, len(lightMatrix), 1)

    return [vehicle_number, halting_number, lgts]
