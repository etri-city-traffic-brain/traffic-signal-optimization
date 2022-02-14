import numpy as np
import copy
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from config import TRAIN_CONFIG

episode_num = 1000
n_step = 500

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    print(tf.reduce_sum(pre_sum, axis=1))
    return tf.reduce_sum(pre_sum, axis=1)

class Actor:
    def __init__(self, name, state_size, action_size, action_min, action_max):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.action = tf.placeholder(tf.float32, [None, action_size])

            inp = tf.layers.dense(self.state, TRAIN_CONFIG['network_size'][0], tf.nn.relu)
            for i in range(len(TRAIN_CONFIG['network_size'])):
                if i!=0:
                    inp = tf.layers.dense(inp, units=TRAIN_CONFIG['network_size'][i], activation=tf.nn.relu)

            self.mu = tf.layers.dense(inp, action_size, tf.tanh)

            self.log_std = tf.get_variable("log_std", initializer=-0.5 * np.ones(action_size, np.float32))
            self.std = tf.exp(self.log_std)
            self.pi = self.mu + tf.random_normal(tf.shape(self.mu)) * self.std
            self.pi = tf.clip_by_value(self.pi, -1, 1)
            self.logp = gaussian_likelihood(self.action, self.mu, self.log_std)
            self.logp_pi = gaussian_likelihood(self.pi, self.mu, self.log_std)


class Critic:
    def __init__(self, name, state_size):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])

            inp = tf.layers.dense(self.state, TRAIN_CONFIG['network_size'][0], tf.nn.relu)
            for i in range(len(TRAIN_CONFIG['network_size'])):
                if i!=0:
                    inp = tf.layers.dense(inp, units=TRAIN_CONFIG['network_size'][i], activation=tf.nn.relu)

            self.value = tf.layers.dense(inp, 1)

            self.v = tf.squeeze(self.value, axis=1)


class PPOAgent:
    def __init__(self, args, state_space, action_space, action_min, action_max, agentID):
        self.state_space = state_space
        self.action_space = action_space

        self._lambda = args._lambda
        self.ppo_eps = args.ppo_eps
        self.learning_rate = args.lr
        self.epoch = args.ppoEpoch
        self.gamma = args.gamma

        self.mode = args.mode
        self.agentID = agentID

        self.actor = Actor("Actor_{}".format(agentID), self.state_space, self.action_space.shape[0], action_min, action_max)
        self.critic = Critic("Critic_{}".format(agentID), self.state_space)
        self.critic_int = Critic("Critic_int{}".format(agentID), self.state_space)

        self.adv = tf.placeholder(tf.float32, [None])
        self.ret = tf.placeholder(tf.float32, [None])
        self.ret_int = tf.placeholder(tf.float32, [None])
        self.logp_old = tf.placeholder(tf.float32, [None])

        self.ratio = tf.exp(self.actor.logp - self.logp_old)
        self.min_adv = tf.where(self.adv > 0, (1.0 + self.ppo_eps) * self.adv, (1.0 - self.ppo_eps) * self.adv)
        self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.adv, self.min_adv))
        self.v_loss = tf.reduce_mean((self.ret - self.critic.v) ** 2)
        self.v_int_loss = tf.reduce_mean((self.ret_int - self.critic_int.v) ** 2)

        self.train_actor = tf.train.AdamOptimizer(self.learning_rate).minimize(self.pi_loss)
        self.train_critic = tf.train.AdamOptimizer(self.learning_rate).minimize(self.v_loss + self.v_int_loss)

        self.approx_kl = tf.reduce_mean(self.logp_old - self.actor.logp)
        self.approx_ent = tf.reduce_mean(-self.actor.logp)


    def update(self, state, action, target, adv, logp_old, sess):
        v_loss, kl, ent = 0, 0, 0
        for i in range(self.epoch):
            _, _, sub_v_loss, approx_kl, approx_ent = \
                sess.run([self.train_actor, self.train_critic, self.v_loss, self.approx_kl, self.approx_ent],
                              feed_dict={self.actor.state: state, self.critic.state: state, self.actor.action: action,
                                         self.ret: target, self.adv: adv, self.logp_old: logp_old})
            v_loss += sub_v_loss
            kl += approx_kl
            ent += approx_ent

        return v_loss, kl, ent

    def get_action(self, state, sess):
        if self.mode=='train':
            action, v, logp_pi = sess.run([self.actor.pi, self.critic.v, self.actor.logp_pi],
                                               feed_dict={self.actor.state: state, self.critic.state: state})
        if self.mode=='test':
            action, v, logp_pi = sess.run([self.actor.mu, self.critic.v, self.actor.logp_pi],
                                               feed_dict={self.actor.state: state, self.critic.state: state})

        return action, v, logp_pi

    def get_gaes(self, rewards, dones, values, next_values, normalize):
        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * self._lambda * gaes[t + 1]
        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes, target

    def compute_intrinsic_reward(self, next_state):
        target_next_feature = self.rnd.target(next_state)
        predict_next_feature = self.rnd.predictor(next_state)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()