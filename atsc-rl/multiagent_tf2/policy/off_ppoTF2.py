# ================================================================
#
#  this code was brought from https://github.com/pythonlessons/Reinforcement_Learning
#   - originally written by PyLessons (2020-10-18)
#   - TensorFlow  : 2.3.1
#   - ref. https://pylessons.com/BipedalWalker-v3-PPO
#     from https://github.com/pythonlessons/Reinforcement_Learning/blob/master/BipedalWalker-v3_PPO/BipedalWalker-v3_PPO.py
#
#  modified to make this can be work with traffic simulation(April. 2022)
#
# ================================================================
import argparse
import copy
import gc
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # -1:cpu, 0:first gpu

import gym
import numpy as np
import pickle
import pylab
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Add, BatchNormalization, Layer 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
__OPTIMIZERS_DIC__={ "adadelta" : Adadelta,
                     "adagrad"  : Adagrad,
                     "adam"     : Adam,
                     "adamax"   : Adamax,
                     "ftrl"     : Ftrl,
                     "nadam"    : Nadam,
                     "rmsprop"  : RMSprop,
                     "sgd"      : SGD
                    }
import collections 

USE_TBX = False
if USE_TBX:
    from tensorboardX import SummaryWriter

#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
# tf.compat.v1.disable_eager_execution()  # usually using this for fastest performance
    # if this is SET, tensorboard does not work


#gpus = tf.config.experimental.list_physical_devices('GPU')
#if len(gpus) > 0:
#    print(f'GPUs {gpus}')
#    try:
#        tf.config.experimental.set_memory_growth(gpus[0], True)
#    except RuntimeError:
#        pass


from DebugConfiguration import DBG_OPTIONS



class BN_Tanh_Dense(Layer):

    def __init__(self, units, kernel_regularizer='l2', use_bias=False):
        
        super(BN_Tanh_Dense, self).__init__()
        
        bn_regularizer = tf.keras.regularizers.l2(l=0.00005)
        #bn_regularizer = tf.keras.regularizers.l2(l=0.0005)
        self.batch_norm = BatchNormalization(momentum=0.9, epsilon=1.0e-5, center=True, scale=True, 
                               beta_regularizer=bn_regularizer, # center
                               gamma_regularizer=bn_regularizer)
        
        #self.activation = tf.nn.elu
        self.activation = tf.nn.tanh
        
        kernel_regularizer = tf.keras.regularizers.l2(l=0.00005) if kernel_regularizer == 'l2' else None
        #kernel_regularizer = tf.keras.regularizers.l2(l=0.0005) if kernel_regularizer == 'l2' else None
        self.dense = Dense(units, kernel_initializer='glorot_uniform', use_bias=use_bias,
                            kernel_regularizer=kernel_regularizer)

        # self.dense = Dense(units, kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        #                     kernel_regularizer=kernel_regularizer)

    def call(self, input_tensor, training=False):
        
        y = self.batch_norm(input_tensor, training=training)
        y = self.activation(y)
        y = self.dense(y)
     
        return y
    


class ResBlock(Layer):

    def __init__(self, units, kernel_regularizer='l2', reduction=4):
        
        super(ResBlock, self).__init__()
        
        bottle_neck_filters = int(units/reduction)
        
        self.input_layer = BN_Tanh_Dense(bottle_neck_filters, kernel_regularizer)
        self.middle_layer = BN_Tanh_Dense(bottle_neck_filters, kernel_regularizer)
        self.output_layer = BN_Tanh_Dense(units, kernel_regularizer)
                    

    def call(self, input_tensor, training=False):
        
        y = self.input_layer(input_tensor, training=training)
        y = self.middle_layer(y, training=training)
        y = self.output_layer(y, training=training)
     
        y = tf.add(input_tensor, y)
        
        return y


class ActorOutput(Layer):
    
    def __init__(self, units):
        
        super(ActorOutput, self).__init__()
        
        self.mu = BN_Tanh_Dense(units, kernel_regularizer=None, use_bias=True)
    
        log_std = np.log(0.25) * np.ones(units, dtype=np.float32)
        #log_std = np.log(0.10) * np.ones(units, dtype=np.float32)
        self.log_std = tf.Variable(initial_value=log_std, trainable=True)
        
    def call(self, input_tensor, training=False):
        
        mu = self.mu(input_tensor, training=training)
        mu = tf.nn.tanh(mu)
        return mu

        
class ActorModel:
    '''
    actor model
    '''
    
    def __init__(self, network_layers, input_shape, action_space, lr, optimizer):

        X_input = Input(input_shape)
        self.action_space = action_space

        X = X_input

        
        for i, size in enumerate(network_layers):
            
            if i == 0:
                kernel_regularizer = tf.keras.regularizers.l2(l=0.00005)
                #kernel_regularizer = tf.keras.regularizers.l2(l=0.0005)
                X = Dense(size, activation=tf.nn.tanh, kernel_initializer='glorot_uniform', kernel_regularizer=kernel_regularizer)(X)
                #X = Dense(size, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=kernel_regularizer)(X)
                
            if i == 1: # to match the dimensions of input and skip onnections
                X = BN_Tanh_Dense(size)(X)            
            
            if i >= 1: 
                X = ResBlock(size)(X)

        #output = BN_Tanh_Dense(self.action_space, kernel_regularizer=None, use_bias=True)(X)
        #output = tf.nn.tanh(output)
            
        output_layer = ActorOutput(self.action_space)
        self.log_std = output_layer.log_std
        
        output = output_layer(X)
            
        self.model = Model(inputs=X_input, outputs=output)
        self.model.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))
        # print(self.model.summary())
        
        #self.log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        #self.log_std = np.log(0.25) + np.zeros(self.action_space, dtype=np.float32)
        #self.min_log_std = np.log(0.1) + np.zeros(self.action_space, dtype=np.float32)
        #self.log_std = np.log(0.50) + np.zeros(self.action_space, dtype=np.float32)
        
        #self.log_std = np.log(0.25) + np.zeros(self.action_space, dtype=np.float32)
        #self.min_log_std = np.log(0.05) + np.zeros(self.action_space, dtype=np.float32)
        


    def decay_std(self, decay_factor):
        
        return
        self.log_std = np.log(decay_factor) + self.log_std
        #self.log_std = np.minimum(self.log_std, self.min_log_std)
        self.log_std = np.maximum(self.log_std, self.min_log_std)
        
    def sample_action(self, state):
        
        mu = self.predict(state)
        std = np.exp(self.log_std)
        #std = tf.math.exp(self.log_std)
        
        actions = mu + np.random.normal(size=mu.shape) * std
        logp_t = self.gaussian_likelihood(actions, mu, self.log_std)
        
        #print('sampling action:', mu, std)
        
        return actions, logp_t, mu, std
    
        
    # def ppo_loss_continuous(self, y_true, y_pred):
        
    #     advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:,
    #                                                                                             1 + self.action_space]
    #     LOSS_CLIPPING = 0.2
    #     logp = self.gaussian_likelihood(actions, y_pred, self.log_std)

    #     logp_old_ph = tf.maximum(logp_old_ph, tf.math.log(0.00001))
    #     ratio = K.exp(logp - logp_old_ph)

    #     p1 = ratio * advantages
    #     p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING) * advantages,
    #                   (1.0 - LOSS_CLIPPING) * advantages)  # minimum advantage

    #     actor_loss = -K.mean(K.minimum(p1, p2))

    #     return actor_loss


    def ppo_loss_continuous(self, y_true, y_pred):
        
        #advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:, 1 + self.action_space]
        
        advantages, actions, logp_old_ph = tf.split(y_true, [1, self.action_space, 1], 1)
        
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred, self.log_std)

        #logp_old_ph = tf.maximum(logp_old_ph, tf.math.log(0.00001))
        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING) * advantages,
                      (1.0 - LOSS_CLIPPING) * advantages)  # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss


    # def ppo_loss_continuous(self, y_true, y_pred):
        
    #     #advantages, actions, logp_old_ph = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:,1 + self.action_space]
    #     advantages, actions, logp_old_ph = tf.split(y_true, [1, self.action_space, 1], 1)
        
    #     logp = self.gaussian_likelihood(actions, y_pred, self.log_std)
        
    #     p1 = logp * advantages
    #     #actor_loss = - K.mean(p1)
    #     actor_loss = - tf.reduce_mean(p1)

    #     return actor_loss

    
    def log_likelihood(self, state, action):
        
        mu = self.predict(state)
        logp = self.gaussian_likelihood(action, mu)
        
        return logp
        
    
    def gaussian_likelihood(self, actions, pred, log_std=None):  # for keras custom loss
        
        log_std = self.log_std if log_std is None else log_std
        #log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions - pred) / (K.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + K.log(2 * np.pi))
        
        return K.sum(pre_sum, axis=1, keepdims=True)



    def predict(self, state):
        return self.model.predict(state, verbose=0)




class CriticModel:
    '''
    critic model
    '''
       
    def __init__(self, network_layers, input_shape, action_space, lr, optimizer):
        
        self.num_critics = 5
        
        self.models = []
        for i in range(self.num_critics):
            model = self.buildDNN(network_layers, input_shape, action_space, lr, optimizer)
            self.models.append(model)


    
    def buildDNN(self, network_layers, input_shape, action_space, lr, optimizer):

        network_layers = (1024, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512)
        #X_input = Input(input_shape)
        #old_values = Input(shape=(1,))
        state_input = Input(input_shape)
        action_input = Input(shape=(action_space, ))
        
        state_stream = state_input
        action_stream = action_input


        for i, size in enumerate(network_layers):
            
            if i == 0:
                kernel_regularizer = tf.keras.regularizers.l2(l=0.00005)
                #kernel_regularizer = tf.keras.regularizers.l2(l=0.0005)
                
                state_stream = Dense(size, activation=tf.nn.tanh, kernel_initializer='glorot_uniform', kernel_regularizer=kernel_regularizer)(state_stream)
                action_stream = Dense(size, activation=tf.nn.tanh, kernel_initializer='glorot_uniform', kernel_regularizer=kernel_regularizer)(action_stream)
                
                #V = Dense(size, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=kernel_regularizer)(V)
            if i == 1:
                state_stream  = BN_Tanh_Dense(size)(state_stream)                        
                action_stream = BN_Tanh_Dense(size)(action_stream)                        
                
            #if 1 <= i <= 2: 
            if 1 <= i <= 5: 
                state_stream  = ResBlock(size)(state_stream)
                action_stream  = ResBlock(size)(action_stream)

            #if i == 2: 
            if i == 5: 
                V = tf.concat([state_stream, action_stream], axis=-1)
                V = BN_Tanh_Dense(size)(V)                        

            #if i > 2:
            if i > 5:
                V = ResBlock(size)(V)
                
                
        value = BN_Tanh_Dense(1, kernel_regularizer=None, use_bias=True)(V)


        model = Model(inputs=[state_input, action_input], outputs=value)
        model.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(lr=lr))

        return model
        
    #@tf.autograph.experimental.do_not_convert
    def critic_PPO2_loss(self, y_true, y_pred):
        #y_true, values = y_true[:, :-1], y_true[:,-1] # y_target, y_old
        #y_true, values = tf.split(y_true, [1, 1], 1)

        # LOSS_CLIPPING = 0.2
        # clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
        # v_loss1 = (y_true - clipped_value_loss) ** 2
        # v_loss2 = (y_true - y_pred) ** 2

        # value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
        
        #value_loss = 0.5 * K.mean((y_true - y_pred) ** 2) # standard PPO loss
        value_loss = 0.5 * tf.reduce_mean(tf.square(y_true - y_pred)) # 


        return value_loss



    # def critic_PPO2_loss(self, values):
    #     def loss(y_true, y_pred):
    #         # LOSS_CLIPPING = 0.2
    #         # clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
    #         # v_loss1 = (y_true - clipped_value_loss) ** 2
    #         # v_loss2 = (y_true - y_pred) ** 2

    #         # value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
    #         value_loss = 0.5 * K.mean((y_true - y_pred) ** 2) # standard PPO loss

    #         # LOSS_CLIPPING = 0.2
    #         # clipped_target = K.clip(y_true, values -LOSS_CLIPPING, values + LOSS_CLIPPING)
    #         # v_loss1 = (clipped_target - y_pred) ** 2
    #         # v_loss2 = (y_true - y_pred) ** 2
    #         # value_loss = 0.5 * K.mean(K.minimum(v_loss1, v_loss2))

    #         return value_loss

    #     return loss


    def predict(self, state, action):

        values = 0
        for model in self.models:
            values_ = model.predict([state, action], verbose=0)
            values += values_

        values = values / self.num_critics
         
        return values


        
    def train(self, critic_input, critic_target, epochs=7, batch_size=1024, verbose=0, shuffle=True):

        total_loss = 0
        for model in self.models:        
            loss = model.fit(critic_input, critic_target, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=shuffle)
            loss = np.asarray(list(loss.history.values())[0])
            total_loss += loss
         
        mean_loss = total_loss / self.num_critics
        
        return mean_loss.tolist()
            


    def load_weights(self, fn_prefix, num_id):

        for i, model in enumerate(self.models):
            model.load_weights(f"{fn_prefix}_{num_id}_critic_{i}.h5")


    def save_weights(self, fn_prefix, num_id):
        
        for i, model in enumerate(self.models):
            model.save_weights(f"{fn_prefix}_{num_id}_critic_{i}.h5")
            


class ReplayMemory:
    '''
    replay memory
    '''
    def __init__(self, max_size, forget_ratio):
        '''
        constructor
        :param max_size: replay memory size
        :param forget_ratio: ratio of forget
        '''
        
        #max_size = 1024
        #max_size = 512
        self.max_size = max_size
        self.num_delete = max_size * forget_ratio
        self.states = collections.deque(maxlen=max_size) #self.states = []
        self.actions = collections.deque(maxlen=max_size) #self.actions = []
        self.rewards = collections.deque(maxlen=max_size) #self.rewards = []
        self.next_states = collections.deque(maxlen=max_size) #self.next_states = []
        self.dones = collections.deque(maxlen=max_size) #self.dones = []
        self.logp_ts = collections.deque(maxlen=max_size) #self.logp_ts = []

        if DBG_OPTIONS.NewModelUpdate:
            self.replay_size = self.max_size * (1.0 - forget_ratio)

    def getSize(self):
        '''
        returns the size of replay memory
        :return:
        '''
        return len(self.states)



    def clear(self):
        '''
        clear replay memory : remove all stored experience
        :return:
        '''
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.logp_ts.clear()



    def reset(self, state, action, reward, next_state, done, logp_t):
        '''
        reset replay memory
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :param logp_t:
        :return:
        '''
        self.clear()
        self.states.append(state) # self.states = [state]
        self.actions.append(action) #self.actions = [action]
        self.rewards.append(reward) #self.rewards = [reward]
        self.next_states.append(next_state) #self.next_states = [next_state]
        self.dones.append(done) #self.dones = [done]
        self.logp_ts.append(logp_t) #self.logp_ts = [logp_t]



    def forget(self):
        '''
        forget certain amount of experience
        :return:
        '''
        # nrc = np.random.choice(range(self.max_size), int(self.num_delete), replace=False)
        # self.states = np.delete(self.states, nrc, axis=0).tolist()
        # self.actions = np.delete(self.actions, nrc, axis=0).tolist()
        # self.rewards = np.delete(self.rewards, nrc, axis=0).tolist()
        # self.next_states = np.delete(self.next_states, nrc, axis=0).tolist()
        # self.dones = np.delete(self.dones, nrc, axis=0).tolist()
        # self.logp_ts = np.delete(self.logp_ts, nrc, axis=0).tolist()

        return


    # if DBG_OPTIONS.NewModelUpdate:
    #     def chooseExperienceToReplay(self):
    #         c_states, c_actions, c_rewards, c_dones, c_next_states, c_logp_ts = [], [], [], [], [], []

    #         cur_mem_size = self.getSize()

    #         if cur_mem_size <= self.replay_size:
    #             c_states = self.states
    #             c_actions = self.actions
    #             c_rewards = self.rewards
    #             c_dones = self.dones
    #             c_next_states = self.next_states
    #             c_logp_ts = self.logp_ts
    #         else:
    #             nrc = np.random.choice(range(cur_mem_size), int(self.replay_size), replace=False)
    #             for i in nrc:
    #                 c_states.append(self.states[i])
    #                 c_actions.append(self.actions[i])
    #                 c_rewards.append(self.rewards[i])
    #                 c_dones.append(self.dones[i])
    #                 c_next_states.append(self.next_states[i])
    #                 c_logp_ts.append(self.logp_ts[i])

    #         return c_states, c_actions, c_rewards, c_dones, c_next_states, c_logp_ts


    if DBG_OPTIONS.NewModelUpdate:
        def chooseExperienceToReplay(self):
            c_states, c_actions, c_rewards, c_dones, c_next_states, c_logp_ts = [], [], [], [], [], []

            c_states = self.states
            c_actions = self.actions
            c_rewards = self.rewards
            c_dones = self.dones
            c_next_states = self.next_states
            c_logp_ts = self.logp_ts

            return c_states, c_actions, c_rewards, c_dones, c_next_states, c_logp_ts





    def store(self, state, action, reward, next_state, done, logp_t):
        '''
        store experience
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :param logp_t:
        :return:
        '''
        # if self.getSize() >= self.max_size:
        #     self.forget()

        # print('Store')
        # print('state')
        # print(state)
        # print('next_state')
        # print(next_state)
        
        # self.states = np.r_[self.states, [state]]
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.logp_ts.append(logp_t)



    def objectDump(self, fn):
        '''
        dump the contents of ReplayMemory into file
        :param fn: file name
        :return:
        '''
        with open(fn, 'wb') as file:
            pickle.dump(self.max_size, file)
            pickle.dump(self.num_delete, file)
            pickle.dump(self.states, file)
            pickle.dump(self.actions, file)
            pickle.dump(self.rewards, file)
            pickle.dump(self.next_states, file)
            pickle.dump(self.dones, file)
            pickle.dump(self.logp_ts, file)



    def objectLoad(self, fn):
        '''
        load the contents of replay memory from a given filw
        :param fn: file name
        :return:
        '''
        with open(fn, 'rb') as file:
            self.max_size = pickle.load(file)  # max_size
            self.num_delete = pickle.load(file) # max_size * forget_ratio
            self.states = pickle.load(file)
            self.actions = pickle.load(file)
            self.rewards = pickle.load(file)
            self.next_states = pickle.load(file)
            self.dones = pickle.load(file)
            self.logp_ts = pickle.load(file)





def testReplayMemory():
    '''
    test objectDump()/objectLoad() in ReplayMemory
    :return:
    '''
    max_size = 10
    forget_ratio = 0.8
    fn = "foo.rm"
    rm1 = ReplayMemory(max_size, forget_ratio)
    for i in range(10):
        state = [i]*5
        action = [1,2,3]
        reward = 0.5
        next_state = [i+1]*5
        done = False
        logp_t = i
        rm1.store(state, action, reward, next_state, done, logp_t)
    rm1.objectDump(fn)

    rm2 = ReplayMemory(max_size+1, forget_ratio+1)
    rm2.objectLoad(fn)

    assert rm1.max_size == rm2.max_size, print("error not equal max_size")
    for i in range(10):
        print(f"rm1.states[{i}]={rm1.states[i]} rm2.states[{i}]={rm2.states[i]}")
        assert rm1.states[i] == rm2.states[i], print(f"error not equal states{i}")
        assert rm1.next_states[i] == rm2.next_states[i], print(f"error not equal next_states{i}")





class PPOAgentTF2:
    '''
    PPO agent
    '''
    def __init__(self, env_name, config, action_size=0, state_size=0, id=""):
        '''
        constructor
        :param env_name:
        :param config:
        :param action_size:
        :param state_size:
        :param id: target identifier managed by this PPO agent
        '''
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        self.action_size = action_size
        self.state_size = state_size
        self.id = id

        self.is_train = config["is_train"]
        self.episode = 0  # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0  # when average score is above 0 model will be saved

        self.network_layers = config["network_layers"]

        self.alr = config["actor_lr"] # 0.00025
        self.clr = config["critic_lr"] # 0.00025
        self.epochs = config["ppo_epoch"] # 10  # training epochs
                
        self.shuffle = True
        self.optimizer = __OPTIMIZERS_DIC__[config["optimizer"].lower()] # Adam

        # used in getGae()
        self.gamma = config["gamma"]  # 0.99
        self.lamda = config["lambda"]  # 0.90

        max_size = config["memory_size"]  # default 512
        forget_ratio = config["forget_ratio"]  # 0.1
        self.memory = ReplayMemory(max_size, forget_ratio)
        self.replay_count = 0

        ### USE_EXPLORATION_EPSILON:
        self.epsilon = config["epsilon"]  # 1 if train, otherwise(test) 0
        self.epsilon_min = config["epsilon_min"]  # 0.1 if train, otherwise(test) 0
        #self.epsilon_decay = config["epsilon_decay"]  # 0.999
        self.epsilon_decay = 0.995

        if USE_TBX :
            self.writer = SummaryWriter(comment="_" + self.env_name + "_" + self.optimizer.__name__ + "_" + str(self.lr))

        # Instantiate plot memory : only used in self.PlotModel()
        self.scores_, self.episodes_, self.average_ = [], [], []  # used in matplotlib plots


        # Create Actor-Critic network models
        self.actor = ActorModel(self.network_layers, input_shape=self.state_size, action_space=self.action_size,
                                lr=self.alr, optimizer=self.optimizer)
        self.critic = CriticModel(self.network_layers, input_shape=self.state_size, action_space=self.action_size,
                                  lr=self.clr, optimizer=self.optimizer)


        self.actor_name = f"{self.id}_PPO_Actor.h5"
        self.critic_name = f"{self.id}_PPO_Critic.h5"
        # self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

        self.init_training = True
        self.actor_history = []
        self.critic_history = []

    def act(self, state):

        #print(f'##### self.std={self.std}  self.log_std={self.log_std}')

        ### USE_EXPLORATION_EPSILON:
        return self.actV3(state)
        # return self.actV2(state)
        # return self.actV1(state)


    def actV1(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.actor.predict(state)

        if self.is_train :
            low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
            action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
            action = np.clip(action, low, high)

            logp_t = self.gaussian_likelihood(action, pred, self.log_std)
        else:
            action = pred
            logp_t = [0]  # when it is not train, this value is not used. so I set dummy value

        return action, logp_t


    def actV2(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.actor.predict(state)

        self.epsilon *= self.epsilon_decay
        # print("act epsilon {}".format(self.epsilon))
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
            action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
            action = np.clip(action, low, high)

            logp_t = self.gaussian_likelihood(action, pred, self.log_std)
        else:
            action = pred
            logp_t = [0]  # when it is not train, this value is not used. so I set dummy value

        return action, logp_t


    def sample_action(self, state):
        print('action sampling')
        action, logp_t, mu, std = self.actor.sample_action(state)
        return action, logp_t, mu, std
    
    def predict_action(self, state):

        print('action prediction')
        action = self.actor.predict(state)
        logp_t = [0]  # when it is not train, this value is not used. so I set dummy value
        mu = action
        std = 0
            
        return action, logp_t, mu, std

        
    def actV3(self, state):

        if self.is_train :

            #self.actor.decay_std(self.epsilon_decay)
            print('action sampling, train')
            action, logp_t, mu, std = self.actor.sample_action(state)
            
            # We should clip the action value when we input it to the simulator.
            # I leave it as it was, to keep simulation working.
            #low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
            #action = np.clip(action, low, high) 

        else:
            print('action prediction, test')
            action = self.actor.predict(state)
            logp_t = [0]  # when it is not train, this value is not used. so I set dummy value
            mu = 0
            std = 0
            
        return action, logp_t, mu, std
    

    def action(self, state, sampling=True):

        if sampling :

            #self.actor.decay_std(self.epsilon_decay)
            print('action sampling, train')
            action, logp_t, mu, std = self.actor.sample_action(state)
            
            # We should clip the action value when we input it to the simulator.
            # I leave it as it was, to keep simulation working.
            #low, high = -1.0, 1.0  # -1 and 1 are boundaries of tanh
            #action = np.clip(action, low, high) 

        else:
            print('action prediction, test')
            action = self.actor.predict(state)
            logp_t = [0]  # when it is not train, this value is not used. so I set dummy value
            mu = 0
            std = 0
            
        return action, logp_t, mu, std
    

    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action - pred) / (np.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return np.sum(pre_sum, axis=1)



    def discount_rewards(self, reward):  # gaes is better... currently not used..
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99  # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8)  # divide by standard deviation
        return discounted_r


    # https://github.com/tensorflow/agents/blob/01c5c40a0229a6745f1fb76a851abc14ad136479/tf_agents/utils/value_ops.py#L98
    # def get_gaes(self, rewards, dones, values, next_values, normalize=True):
    #     #print(len(rewards), len(dones), len(values), len(next_values))
                
    #     deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    #     deltas = np.stack(deltas)
    #     gaes = copy.deepcopy(deltas)
    #     for t in reversed(range(len(deltas) - 1)):
    #         gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * self.lamda * gaes[t + 1]
    #         print(t)
    #     #print(len(gaes))
    #     #print(gaes)
    #     target = gaes + values
    #     if normalize:
    #         gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        
    #     return np.vstack(gaes), np.vstack(target)

    def get_gaes(self, rewards, dones, values, next_values, normalize=True):


        targets = [r + self.gamma * (1 - d) * nv for r, d, nv in zip(rewards, dones, next_values)]
        advantages = [ t - v for t, v in zip(targets, values)]
        if normalize:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                
        return np.vstack(advantages), np.vstack(targets)


    def replay(self):
        # return self.replayOrgSucc()
        return self.replayNew()

    def replayOrgSucc(self):

        if not self.is_train:  # no need to replay if it is not the target of training
            return

        states = self.memory.states
        actions = self.memory.actions
        rewards = self.memory.rewards
        dones = self.memory.dones
        next_states = self.memory.next_states
        logp_ts = self.memory.logp_ts

        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        # Compute discounted rewards and advantages
        # discounted_r = self.discount_rewards(rewards)
        # advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, actions, logp_ts])

        # training Actor and Critic networks
        a_loss = self.actor.model.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.critic.model.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        
        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        pred = self.actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        if 1:
            # from TSOUtil import total_size
            # num_entry = len(states)
            # sz_states = total_size(states, verbose=False)
            # sz_next_states = total_size(next_states, verbose=False)
            # sz_actions = total_size(actions, verbose=False)
            # sz_logp_ts = total_size(logp_ts, verbose=False)
            # sz_y_true= total_size(y_true)
            #
            # print(f"num_entry={num_entry} sz_states={sz_states} sz_n_states={sz_next_states} sz_act={sz_actions} sz_logp_ts={sz_logp_ts} sz_y_true={sz_y_true}")

            del states
            del next_states
            del actions
            del logp_ts
            del y_true
            gc.collect()

        if USE_TBX:
            self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
            self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
            self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
            self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)

        self.replay_count += 1



    def evaluate_state(self, states):
        #actions = self.actor.sample_action(states)
        #samplings = 100
        samplings = 10
        values = 0
        for i in range(samplings):
            actions, logp , mu, std = self.actor.sample_action(states)
            values = values + self.critic.predict(states, actions)
        
        return values / samplings

    def evaluate_state_action(self, states, actions):
        #actions = self.actor.sample_action(states)
        values = self.critic.predict(states, actions)
        
        return values 

    
    def compute_target_value(self, rewards, dones, next_values):

        if self.init_training:
            targets = [r + 0 * nv for r, nv in zip(rewards, next_values)]
        else: 
            targets = [r + self.gamma * (1 - d) * nv for r, d, nv in zip(rewards, dones, next_values)]
        
        # normalize target values
        targets = np.vstack(targets)
        #targets = (targets - np.mean(targets)) / (np.std(targets) + 1e-8)
        
        return targets
    
    
    def compute_advantage(self, states, actions):
        
        state_action_values = self.critic.predict(states, actions) #q
        state_values = self.evaluate_state(states) # v
        advantages = state_action_values - state_values

        return advantages
    
    def normalize_advantage(self, advantages):


        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        return advantages
        

    def augment_play(self, state, samplings=20):
        
        states = []
        actions = []
        log_probs = []
        state_action_values = []
        
        state_value = 0 # v
        
        for i in range(samplings):
            
            action, log_prob, mu, std = self.actor.sample_action(state)
            state_action_value = self.critic.predict(state, action) #q
            state_value = state_value + state_action_value
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            state_action_values.append(state_action_value)    
        
        state_value = state_value / samplings
        advantages = [q - state_value for q in state_action_values]
        
        states = np.vstack(states)
        actions = np.vstack(actions)
        log_probs = np.vstack(log_probs)
        advantages = np.vstack(advantages)
        
        #advantages = self.normalize_advantage(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return states, actions, log_probs, advantages
    
    
   
    
    def replayNew(self):

        if not self.is_train:  # no need to replay if it is not the target of training
            return


        if DBG_OPTIONS.NewModelUpdate:
            states, actions, rewards, dones, next_states, logp_ts = self.memory.chooseExperienceToReplay()
            print("got replay items in replayNew() at PPOAgentTF2")
        else:
            states = self.memory.states
            actions = self.memory.actions
            rewards = self.memory.rewards
            dones = self.memory.dones
            next_states = self.memory.next_states
            logp_ts = self.memory.logp_ts

        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        #logp_ts = np.vstack(logp_ts)
        
        
        print('Fitting Critics...')
        for value_update in range(7):        
            #values = self.critic.predict(states, actions)
            next_values = self.evaluate_state(next_states)
            target = self.compute_target_value(rewards, dones, np.squeeze(next_values))

            samples = target.shape[0]
            #print('training samples for a critic', samples)
            critic_input = [states, actions]
            #critic_target = np.hstack([target, values])
            critic_target = target
           
            c_loss = self.critic.train(critic_input, critic_target, epochs=7, batch_size=samples, verbose=0, shuffle=True)
            print('Critic loss:', c_loss)
            self.init_training = False
            
        self.critic_history.append(c_loss[-1])
        print('Critic loss:', self.critic_history[-10:])
        
        #self.init_training = False

        print('Augmenting training data for actor...')
        states, actions, logp_ts, advantages = self.augment_play(states)
        #states, actions, logp_ts, advantages = self.augment_play(states, actions, rewards)
        samples = len(logp_ts)
        print('training samples for an actor', samples)

        actor_input = states
        actor_target = np.hstack([advantages, actions, logp_ts])

        print('Trianing actor...')
        a_loss = self.actor.model.fit(actor_input, actor_target, epochs=5, verbose=0, batch_size=samples, shuffle=True)
        a_loss = list(a_loss.history.values())[0]
        self.actor_history.append(a_loss[-1])
        print('Actor loss:', a_loss)
        print('Actor loss:', self.actor_history[-10:])

         # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        pred = self.actor.predict(states)
        #log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        log_std = self.actor.log_std
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)
        
        self.actor.decay_std(self.epsilon_decay)
        #self.memory.clear()
        
        # if 1:
        #     # from TSOUtil import total_size
        #     # num_entry = len(states)
        #     # sz_states = total_size(states, verbose=False)
        #     # sz_next_states = total_size(next_states, verbose=False)
        #     # sz_actions = total_size(actions, verbose=False)
        #     # sz_logp_ts = total_size(logp_ts, verbose=False)
        #     # sz_y_true= total_size(y_true)
        #     #
        #     # print(f"num_entry={num_entry} sz_states={sz_states} sz_n_states={sz_next_states} sz_act={sz_actions} sz_logp_ts={sz_logp_ts} sz_y_true={sz_y_true}")

        #     del states
        #     del next_states
        #     del actions
        #     del logp_ts
        #     #del y_true
        #     gc.collect()

        if USE_TBX:
            self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
            self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
            self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
            self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)

        self.replay_count += 1




    def loadModel(self, fn_prefix):

        self.actor.model.load_weights(f"{fn_prefix}_{self.id}_actor.h5")
        self.critic.load_weights(fn_prefix, self.id)


    def saveModel(self, fn_prefix):

        self.actor.model.save_weights(f"{fn_prefix}_{self.id}_actor.h5")
        self.critic.save_weights(fn_prefix, self.id)
        


    def dumpReplayMemory(self, fn):
        '''
        dump the contents of replay memory into file
        :param fn:
        :return:
        '''
        self.memory.objectDump(fn)



    def loadReplayMemory(self, fn):
        '''
        load the contents of replay memory from a given file
        :param fn:
        :return:
        '''
        self.memory.objectLoad(fn)



    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)

    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":  # much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name + ".png")
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average and save:
            self.max_average = self.average_[-1]
            self.saveModel(self.actor_name)
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            # self.lr *= 0.99
            # K.set_value(self.actor.model.optimizer.learning_rate, self.lr)
            # K.set_value(self.critic.model.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING




## need to create one for each policy
def makePPOConfig(args):
    '''
    make configuration dictionary for PPO
    :param args: argument
    :return:
    '''

    cfg = {}

    cfg["state"] = args.state
    cfg["action"] = args.action
    cfg["reward"] = args.reward_func

    # cfg["lr"] = args.lr  # 0.005
    cfg["gamma"] = args.gamma  # 0.99
    cfg["lambda"] = args._lambda  # 0.95
    cfg["actor_lr"] = args.a_lr  # 0.005
    cfg["critic_lr"] = args.c_lr  # 0.005
    cfg["ppo_epoch"] = args.ppo_epoch  # 10
    cfg["ppo_eps"] = args.ppo_eps  # 0.1  # used for ppoea

    cfg["memory_size"] = args.mem_len
    cfg["forget_ratio"] = args.mem_fr

    cfg["offset_range"] = args.offset_range  # 2
    cfg["control_cycle"] = args.control_cycle  # 5
    cfg["add_time"] = args.add_time  # 2

    # for exploration
    cfg["epsilon"] = args.epsilon  # epsilon for exploration
    cfg["epsilon_min"] = args.epsilon_min  # minimum of epsilon for exploration
    cfg["epsilon_decay"] = args.epsilon_decay  # epsilon decay for exploration
    cfg["epoch_exploration_decay"] = args.epoch_exploration_decay  # epsilon decay for exploration

    cfg["network_layers"] = args.network_size

    cfg["optimizer"] = args.optimizer
    return cfg


## need to create one for each policy
def makePPOProblemVar(conf):
    '''
    make string by concatenating configuration
    this will be used as a prefix of file/path name to store log, model, ...

    :param conf:
    :return:
    '''

    problem_var = ""
    problem_var += "_state_{}".format(conf["state"])
    problem_var += "_action_{}".format(conf["action"])
    problem_var += "_reward_{}".format(conf["reward"])

    #problem_var += "_gamma_{}".format(conf["gamma"])
    #problem_var += "_lambda_{}".format(conf["lambda"])
    problem_var += "_alr_{}".format(conf["actor_lr"])
    problem_var += "_clr_{}".format(conf["critic_lr"])

    problem_var += "_mLen_{}".format(conf["memory_size"])
    #problem_var += "_mFR_{}".format(conf["forget_ratio"])
    #problem_var += "_netSz_{}".format(conf["network_layers"])
    #problem_var += "_offset_range_{}".format(conf["offset_range"])
    problem_var += "_control_cycle_{}".format(conf["control_cycle"])
    
#    if DBG_OPTIONS.AddControlCycleIntoProblemVar:
#        problem_var += "_control_cycle_{}".format(conf["control_cycle"])

    # error message = 'File name too long'
#    if 0:
#        # for exploration
#        problem_var += "_epsilon_{}".format(conf["epsilon"])
#        problem_var += "_epsilon_min_{}".format(conf["epsilon_min"])
#        problem_var += "_epsilon_decay{}".format(conf["epsilon_decay"])
#        problem_var += "_epoch_exploration_decay{}".format(conf["epoch_exploration_decay"])

    # if args.method=='ppornd':
    #     problem_var += "_gammai_{}".format(args.gamma_i)
    #     problem_var += "_rndnetsize_{}".format(TRAIN_CONFIG['rnd_network_size'])
    # if args.method=='ppoea':
    #     problem_var += "_ppo_epoch_{}".format(args.ppo_epoch)
    #     problem_var += "_ppoeps_{}".format(args.ppo_eps)
    # if len(args.target_TL.split(","))==1:
    #     problem_var += "_{}".format(args.target_TL.split(",")[0])
    #
    # if args.action == 'gr' or args.action == 'gro':
    #     problem_var += "_addTime_{}".format(args.add_time)

    return problem_var



#####################################################
###
## belows are to check PPOAgentTF2 is correctly work
##
def run_batch(env, agent, trials):

    EPISODES = 200000  # total episodes to train through all environments
    EPISODES = trials  # total episodes to train through all environments
    Training_batch = 512
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size[0]])

    done, score, SAVING = False, 0, ''
    while True:
        # Instantiate or reset games memory
        agent.memory.clear()

        for t in range(Training_batch):  # 512...
            env.render()
            # Actor picks an action
            action, logp_t = agent.act(state)
            # Retrieve new state, reward, and whether the state is terminal
            next_state, reward, done, _ = env.step(action[0])

            # Memorize (state, next_states, action, reward, done, logp_ts) for training
            agent.memory.store(state, action, reward, np.reshape(next_state, [1, agent.state_size[0]]), done,
                                  logp_t[0])

            # Update current state shape
            state = np.reshape(next_state, [1, agent.state_size[0]])
            score += reward

            if done:
                agent.episode += 1
                average, SAVING = agent.PlotModel(score, agent.episode)
                print("episode: {}/{}, score: {}, average: {:.2f} {}".format(agent.episode, EPISODES, score,
                                                                             average, SAVING))
                if USE_TBX:
                    agent.writer.add_scalar(f'Workers:{1}/score_per_episode', score, agent.episode)
                    agent.writer.add_scalar(f'Workers:{1}/learning_rate', agent.lr, agent.episode)
                    agent.writer.add_scalar(f'Workers:{1}/average_score', average, agent.episode)

                # print("done={} ... current step = {}".format(done, t))
                #     #-- done=True ... current step = 413
                state, done, score, SAVING = env.reset(), False, 0, ''
                state = np.reshape(state, [1, agent.state_size[0]])

        agent.replay2()

        # print("replay done : memsize={} : current step={}".format(self.memory.getSize(), t))
        #     #-- replay done : memsize=512 : current step=511

        if agent.episode >= EPISODES:
            break
    agent.saveModel(agent.actor_name)
    env.close()


def test(env, agent, test_episodes=100):  # evaluate
    agent.loadModel(agent.actor_name)
    for e in range(101):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size[0]])
        done = False
        score = 0
        while not done:
            env.render()
            if 0:
                action = agent.actor.predict(state)[0]
            else:
                action, _ = agent.act(state)
                action = action[0]

            state, reward, done, _ = env.step(action)
            state = np.reshape(state, [1, agent.state_size[0]])
            score += reward
            if done:
                average, SAVING = agent.PlotModel(score, e, save=False)
                print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                break
    env.close()


ORG = False



if __name__ == "__main__":
    # import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='train - RL model training, test - trained model testing')
    parser.add_argument('--trials', type=int, default=1,
                        help='train - RL model training, test - trained model testing')
    args = parser.parse_args()

    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
    env_name = 'BipedalWalker-v3'
    env = gym.make(env_name)

    if 0:
        print("action_shape =", env.action_space.shape)  # (4, )
        print("observation_shape =", env.observation_space.shape)  # (24, 0)
        print("action_space.shape[0]={}".format(env.action_space.shape[0]))  # 4
        print(
            "action_space.n = {} ".format(env.action_space.n))  # AttributeError: 'Box' object has no attribute 'n'


    if 1:
        args.state = 'vdd'
        args.action = 'offset'
        args.reward_func = 'wq'
        args.lr = 0.005  # 0.00025
        args.ppo_epoch = 10

        args.gamma = 0.99
        args._lambda = 0.95 # 0.90
        args.a_lr = 0.005
        args.c_lr = 0.005
        args.ppo_eps = 0.1
        args.mem_len = 10000
        args.mem_fr = 0.9
        args.offset_range = 2
        args.control_cycle = 5
        args.add_time = 2
        # args.optimizer = Adam
        args.optimizer = "Adam"


    config = makePPOConfig(args)

    is_train = True if args.mode == 'train' else False
    config["is_train"] = is_train

    action_size = env.action_space.shape[0]   #  4
    state_size = env.observation_space.shape  # (24, 0)

    agent = PPOAgentTF2(env_name, config, action_size, state_size, "")

    if args.mode == 'train':
        run_batch(env, agent, args.trials)  # train as PPO
    elif args.mode == 'test':
        test(env, agent)
    # agent.run_multiprocesses(num_worker = 16)  # train PPO multiprocessed (fastest)
