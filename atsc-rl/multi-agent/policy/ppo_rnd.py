
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from config import TRAIN_CONFIG

import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import itertools

import sklearn.pipeline
import sklearn.preprocessing

# Approximates feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform.
# https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html
from sklearn.kernel_approximation import RBFSampler

class RunningStats(object):
    # This class which computes global stats is adapted & modified from:
    # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        #self.std = np.sqrt(np.maximum(self.var, 1e-2))
        self.count = batch_count + self.count

class PPORNDAgent(object):
    def __init__(self, args, state_space, action_space, action_min, action_max, agentID):
        self.s = tf.placeholder(tf.float32, [None, state_space], 'state')
        self.s_ = tf.placeholder(tf.float32, [None, state_space], 'state_')
        self.action_space = action_space.shape[0]
        self.epsilon = 0.1

        self.ext_r_coeff = 2  # extrinsic reward coefficient
        self.c_loss_coeff = 0.5  # coefficient for total critic loss

        self.encode_features = 64
        self.ENTROPY_BETA = 0.001

        self.s_CLIP = 10  # clip for state buffer
        self.next_s_CLIP = 5  # clip for next state's buffer
        self.r_CLIP = 1  # clip for extrinsic reward, note that intrinsic rewards are not clip.

        # Hyper parameters for computation of TD lambda return and policy advantage using GAE
        self.GAMMA = 0.999  # 0.95 #0.95 # discount factor for extrinsic reward
        self.GAMMA_i = 0.99  # 0.95 # discount factor for intrinsic reward
        self.lamda = 0.95

        self.args = args
        # RND
        with tf.variable_scope('RND'):
          with tf.variable_scope('target'):
            r_w = tf.random_normal_initializer()
            # Fixed target network encodes state to features
            # Network randomly initialized once but never trained, params remain fixed, trainable=False
            # self.target_out = tf.layers.dense(self.s_, self.encode_features, kernel_initializer = r_w, name='target_out', trainable=False)
            # hidden_layer = tf.layers.dense(self.s_, num_hidden, tf.nn.relu, kernel_initializer = r_w, name='t_hidden', trainable=False)
            inp = tf.layers.dense(self.s_, TRAIN_CONFIG['rnd_network_size'][0], tf.nn.relu)
            for i in range(len(TRAIN_CONFIG['rnd_network_size'])):
                if i != 0:
                    inp = tf.layers.dense(inp, units=TRAIN_CONFIG['network_size'][i], activation=tf.nn.relu)
            # self.target_out = tf.layers.dense(inp, self.encode_features, kernel_initializer = r_w, name='target_out', trainable=False)
            self.target_out = tf.layers.dense(inp, self.encode_features, name='target_out', trainable=False)
        # predictor network
          with tf.variable_scope('predictor'):
            #p_w = tf.zeros_initializer()
            p_w = tf.random_normal_initializer()
            #p_w = tf.glorot_uniform_initializer(seed=tf_operation_level_seed+1)
            # self.predictor_out = tf.layers.dense(self.s_, self.encode_features, kernel_initializer = p_w, name='predictor_out', trainable=True)
            # hidden_layer = tf.layers.dense(self.s_, num_hidden, tf.nn.relu, kernel_initializer = p_w, name='p_hidden', trainable=True)
            inp = tf.layers.dense(self.s_, TRAIN_CONFIG['rnd_network_size'][0], tf.nn.relu)
            for i in range(len(TRAIN_CONFIG['rnd_network_size'])):
                if i != 0:
                    inp = tf.layers.dense(inp, units=TRAIN_CONFIG['network_size'][i], activation=tf.nn.relu)
            # self.predictor_out = tf.layers.dense(inp, self.encode_features, kernel_initializer = p_w, name='predictor_out', trainable=True)
            self.predictor_out = tf.layers.dense(inp, self.encode_features, name='predictor_out', trainable=True)
            # self.predictor_loss is also the intrinsic reward
            self.predictor_loss = tf.reduce_sum(tf.square(self.target_out - self.predictor_out), axis=1)

        with tf.variable_scope('PPO'):
          # critic
          with tf.variable_scope('critic'):
            #c_w = tf.zeros_initializer()
            c_w = tf.random_normal_initializer()
            #c_w = tf.glorot_uniform_initializer(seed=tf_operation_level_seed+2)
            with tf.variable_scope('critic_extrinsic'):
                # critic network for extrinsic reward
                # self.v = tf.layers.dense(self.s, 1, kernel_initializer = c_w, name='val', trainable=True)
                # hidden_layer = tf.layers.dense(self.s, num_hidden, tf.nn.relu, kernel_initializer = c_w, name='c_e_hidden', trainable=True)
                inp = tf.layers.dense(self.s, TRAIN_CONFIG['network_size'][0], tf.nn.relu)
                for i in range(len(TRAIN_CONFIG['network_size'])):
                    if i != 0:
                        inp = tf.layers.dense(inp, units=TRAIN_CONFIG['network_size'][i], activation=tf.nn.relu)
                # self.v = tf.layers.dense(inp, 1, kernel_initializer = c_w, name='val', trainable=True)
                self.v = tf.layers.dense(inp, 1, name='val', trainable=True)
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))
            with tf.variable_scope('critic_intrinsic'):
                # critic network for intrinsic reward
                # self.v_i = tf.layers.dense(self.s, 1, kernel_initializer = c_w, name='val_i', trainable=True)
                # hidden_layer = tf.layers.dense(self.s, num_hidden, tf.nn.relu, kernel_initializer = c_w, name='c_i_hidden', trainable=True)

                inp = tf.layers.dense(self.s, TRAIN_CONFIG['network_size'][0], tf.nn.relu)
                for i in range(len(TRAIN_CONFIG['network_size'])):
                    if i != 0:
                        inp = tf.layers.dense(inp, units=TRAIN_CONFIG['network_size'][i], activation=tf.nn.relu)

                # self.v_i = tf.layers.dense(inp, 1, kernel_initializer = c_w, name='val_i', trainable=True)
                self.v_i = tf.layers.dense(inp, 1, name='val_i', trainable=True)
                self.tfdc_r_i = tf.placeholder(tf.float32, [None, 1], 'discounted_r_i')
                self.advantage_i = self.tfdc_r_i - self.v_i
                self.closs_i = tf.reduce_mean(tf.square(self.advantage_i))
            with tf.variable_scope('total_critic_loss'):
                self.total_closs = self.closs + self.closs_i

          # actor
          with tf.variable_scope('actor'):
            pi, pi_params, mu = self._build_anet('pi', trainable=True)
            oldpi, oldpi_params, old_mu = self._build_anet('oldpi', trainable=False) # trainable=False
            with tf.variable_scope('sample_action'):
                if self.args.mode == 'train':
                    self.sample_op = tf.squeeze(pi.sample(1), axis=0) # choosing action
                elif self.args.mode == 'test':
                    self.sample_op = tf.squeeze(mu, axis=0) # choosing action
            with tf.variable_scope('update_oldpi'):
              self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
            with tf.variable_scope('surrogate_actor_loss'):
              self.tfa = tf.placeholder(tf.float32, [None, self.action_space], 'action')
              self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
              ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
              surr = ratio * self.tfadv
              self.aloss = -tf.reduce_mean(tf.minimum(surr,
                                                      tf.clip_by_value(ratio, 1.-self.epsilon, 1.+self.epsilon)*self.tfadv))

            with tf.variable_scope('entropy'):
              entropy = -tf.reduce_mean(pi.entropy()) # Compute the differential entropy of the multivariate normal.

        with tf.variable_scope('total_loss'):
            self.total_loss = tf.reduce_mean(self.predictor_loss) + self.total_closs * self.c_loss_coeff + self.aloss
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss + entropy * self.ENTROPY_BETA,
                                                                   global_step=tf.train.get_global_step())

    def update(self, s, s_, a, r, r_i, adv, sess):
        sess.run(self.update_oldpi_op)
        [sess.run(self.train_op, {self.s: s, self.tfa: a, self.tfadv: adv, self.tfdc_r: r, self.tfdc_r_i: r_i, self.s_: s_}) for _ in range(self.args.ppoEpoch)]

    def _build_anet(self, name, trainable):
        # tanh range = [-1,1]
        # softplus range = {0,inf}
        with tf.variable_scope(name):
            #a_w = tf.zeros_initializer()
            #a_w = tf.random_normal_initializer(seed=tf_operation_level_seed+3) # can't use random for actor, produces nan action
            a_w = tf.glorot_uniform_initializer()
            # mu = tf.layers.dense(self.s, self.action_space, tf.nn.tanh, kernel_initializer = a_w, name='mu', trainable=trainable)
            # sigma = tf.layers.dense(self.s, self.action_space, tf.nn.softplus, kernel_initializer = a_w, name='sigma', trainable=trainable) + 1e-4
            # hidden_layer = tf.layers.dense(self.s, 1024, tf.nn.relu, kernel_initializer = a_w, name='a_hidden', trainable=trainable)
            # mu = tf.layers.dense(hidden_layer, self.action_space, tf.nn.tanh, kernel_initializer = a_w, name='mu', trainable=trainable)
            # sigma = tf.layers.dense(hidden_layer, self.action_space, tf.nn.softplus, kernel_initializer = a_w, name='sigma', trainable=trainable) + 1e-4

            inp_mu = tf.layers.dense(self.s, TRAIN_CONFIG['network_size'][0], tf.nn.relu)
            for i in range(len(TRAIN_CONFIG['network_size'])):
                if i!=0:
                    inp_mu = tf.layers.dense(inp_mu, units=TRAIN_CONFIG['network_size'][i], activation=tf.nn.relu)
            inp_sigma = tf.layers.dense(self.s, TRAIN_CONFIG['network_size'][0], tf.nn.relu)
            for i in range(len(TRAIN_CONFIG['network_size'])):
                if i!=0:
                    inp_sigma = tf.layers.dense(inp_sigma, units=TRAIN_CONFIG['network_size'][i], activation=tf.nn.relu)

            # mu = tf.layers.dense(inp_mu, self.action_space, tf.nn.tanh, kernel_initializer=a_w, name='mu', trainable=trainable)
            mu = tf.layers.dense(inp_mu, self.action_space, tf.nn.tanh, name='mu', trainable=trainable)
            # sigma = tf.layers.dense(inp_sigma, self.action_space, tf.nn.softplus, kernel_initializer=a_w, name='sigma', trainable=trainable) + 1e-4
            sigma = tf.layers.dense(inp_sigma, self.action_space, tf.nn.softplus, name='sigma', trainable=trainable) + 1e-4

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, mu

    def choose_action(self, s, sess):
        if self.args.mode == 'train':
            a = sess.run(self.sample_op, {self.s: s})[0]
            # print(s, a)
        elif self.args.mode == 'test':
            a = sess.run(self.sample_op, {self.s: s})
            # print(s, a)
        return np.clip(a, -1, 1)

    def get_v(self, s, sess):
        return sess.run(self.v, {self.s: s})[0, 0]

    def get_v_i(self, s, sess):
        return sess.run(self.v_i, {self.s: s})[0, 0]

    def intrinsic_r(self, s_, sess):
        return sess.run(self.predictor_loss, {self.s_: s_})

    def add_vtarg_and_adv(self, R, done, V, v_s_, gamma, lam):
        # This function is adapted & modified from:
        # https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py
        # Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        # last element is only used for last vtarg, but we already zeroed it if last new = 1
        done = np.append(done, 0)
        V_plus = np.append(V, v_s_)
        T = len(R)
        adv = gaelam = np.empty(T, 'float32')
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-done[t+1]
            delta = R[t] + gamma * V_plus[t+1] * nonterminal - V_plus[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        tdlamret = np.vstack(adv) + V
        return tdlamret, adv # tdlamret is critic_target or Qs
