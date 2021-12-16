import numpy as np
import random
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from time import time
from keras.callbacks import TensorBoard, LearningRateScheduler, Callback

import keras.backend as K
from collections import deque

class DQN:
    def __init__(self, env, state_space, action_space, epsilon=1, epsilon_min=0.1):
        self.env = env
        self.memory = deque(maxlen=100000)

        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.01
        self.tau = 0.125
        # self.tau = 0.05

        self.batch_size = 32

        self.state_space = state_space
        self.action_space = action_space

        self.model = self.create_model()
        self.target_model = self.create_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def create_model(self):
        model = Sequential()
        state_shape = self.state_space
        # model.add(Dense(1000, input_dim=state_shape, activation="relu"))
        # model.add(Dense(800, activation="relu"))
        # model.add(Dense(600, activation="relu"))
        # model.add(Dense(400, activation="relu"))
        # model.add(Dense(200, activation="relu"))
        model.add(Dense(self.action_space*8, input_dim=state_shape, activation="relu"))
        model.add(Dense(self.action_space*8, activation="relu"))
        model.add(Dense(self.action_space*4, activation="relu"))
        model.add(Dense(self.action_space*4, activation="relu"))
        model.add(Dense(self.action_space*2, activation="relu"))
        model.add(Dense(self.action_space*2, activation="relu"))
        model.add(Dense(self.action_space))

        # model.compile(loss="mean_squared_error",
        #               optimizer=Adam(lr=self.learning_rate))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def act(self, state):
        self.epsilon *= self.epsilon_decay
        # print("act epsilon {}".format(self.epsilon))
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.sample(list(range(self.action_space)),1)[0]
        # print(state.reshape(self.state_space, 1))
        # print(state)
        return np.argmax(self.model.predict(state.reshape(1,self.state_space))[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)


        update_input = np.zeros((self.batch_size, self.state_space))
        update_target = np.zeros((self.batch_size, self.state_space))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = samples[i][0]
            action.append(samples[i][1])
            reward.append(samples[i][2])
            update_target[i] = samples[i][3]
            done.append(samples[i][4])

        target = self.target_model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_val[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

        # for sample in samples:
        #     state, action, reward, new_state, done = sample
        #     target = self.target_model.predict(state.reshape(1,self.state_space))
        #     if done:
        #         target[0][action] = reward
        #     else:
        #         Q_future = max(self.target_model.predict(new_state.reshape(1,self.state_space))[0])
        #         target[0][action] = reward + Q_future * self.gamma
        #     self.model.fit(state.reshape(1,self.state_space), target, epochs=1, verbose=0)
        #     # print("optimizer lr ", round(self.model.optimizer.lr.numpy(), 5))


    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self, fn):
        self.model.load_weights(fn)