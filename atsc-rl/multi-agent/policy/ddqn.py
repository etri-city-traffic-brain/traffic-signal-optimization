import numpy as np
import random
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# from time import time
# from keras.callbacks import TensorBoard, LearningRateScheduler, Callback

import keras.backend as K
from collections import deque

from config import TRAIN_CONFIG

class DDQN:
    def __init__(self, args, env, state_space, action_space, epsilon=1, epsilon_min=0.1):
        self.env = env
        self.memory = deque(maxlen=TRAIN_CONFIG['replay_size'])

        self.gamma = args.gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 0.999
        self.learning_rate = TRAIN_CONFIG['lr']
        self.tau = args.tau
        # self.tau = 0.05

        self.batch_size = TRAIN_CONFIG['batch_size']

        self.state_space = state_space
        self.action_space = action_space

        self.model = self.create_model()
        self.target_model = self.create_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def create_model(self):
        model = Sequential()
        state_shape = self.state_space
        print(TRAIN_CONFIG['network_size'])
        for ns in range(len(TRAIN_CONFIG['network_size'])):
            if ns==0:
                model.add(Dense(TRAIN_CONFIG['network_size'][ns], input_dim=state_shape, activation="relu"))
            else:
                model.add(Dense(TRAIN_CONFIG['network_size'][ns], activation="relu"))
        # model.add(Dense(self.action_space*100, input_dim=state_shape, activation="relu"))
        # model.add(Dense(self.action_space*80, activation="relu"))
        # model.add(Dense(self.action_space*60, activation="relu"))
        # model.add(Dense(self.action_space*40, activation="relu"))
        # model.add(Dense(self.action_space*20, activation="relu"))
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
            return random.sample(list(range(self.action_space)), 1)[0]
        # print(state.reshape(self.state_space, 1))
        # print(state)
        return np.argmax(self.model.predict(state.reshape(1, self.state_space))[0])

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

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
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
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    ## ref. https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko
    ## https://soundprovider.tistory.com/entry/tensorflow-20-modelsave-modelsaveweights-%EC%B0%A8%EC%9D%B4
    def save_model(self, fn):
        if 1:
            self.model.save_weights(fn)
        else:
            self.model.save(fn)


    def load_model(self, fn):
        if 1:
            self.model.load_weights(fn)
        else:
            self.model = tf.keras.models.load_model(fn)

