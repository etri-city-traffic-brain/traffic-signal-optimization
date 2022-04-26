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
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # -1:cpu, 0:first gpu

import gym
import pylab
import numpy as np
import tensorflow as tf

USE_TBX = False
if USE_TBX:
    from tensorboardX import SummaryWriter

tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
# tf.compat.v1.disable_eager_execution()  # usually using this for fastest performance
    # if this is SET, tensorboard does not work

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras import backend as K
import copy


gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


class ActorModel:
    '''
    actor model
    '''

    def __init__(self, network_layers, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        X = X_input

        for size in network_layers:
            X = Dense(size, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)

        output = Dense(self.action_space, activation="tanh")(X)

        self.model = Model(inputs=X_input, outputs=output)
        self.model.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))
        # print(self.model.summary())

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:,
                                                                                               1 + self.action_space]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING) * advantages,
                      (1.0 - LOSS_CLIPPING) * advantages)  # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred):  # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions - pred) / (K.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + K.log(2 * np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.model.predict(state)


class CriticModel:
    '''
    critic model
    '''
    def __init__(self, network_layers, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        V = X_input

        for size in network_layers:
            V = Dense(size, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)

        value = Dense(1, activation=None)(V)

        self.model = Model(inputs=[X_input, old_values], outputs=value)
        self.model.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2

            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            # value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss

            return value_loss

        return loss

    def predict(self, state):
        return self.model.predict([state, np.zeros((state.shape[0], 1))])



class ReplayMemory :
    '''
    replay memory
    '''
    def __init__(self, max_size, forget_ratio):
        '''
        constructor
        :param max_size: replay memory size
        :param forget_ratio: ratio of forget
        '''
        self.max_size = max_size
        self.num_delete = max_size * forget_ratio
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.logp_ts = []

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
        self.states = [state]
        self.actions = [action]
        self.rewards = [reward]
        self.next_states = [next_state]
        self.dones = [done]
        self.logp_ts = [logp_t]


    def forget(self):
        '''
        forget certain amount of experience
        :return:
        '''
        nrc = np.random.choice(range(self.max_size), int(self.num_delete), replace=False)
        self.states = np.delete(self.states, nrc, axis=0).tolist()
        self.actions = np.delete(self.actions, nrc, axis=0).tolist()
        self.rewards = np.delete(self.rewards, nrc, axis=0).tolist()
        self.next_states = np.delete(self.next_states, nrc, axis=0).tolist()
        self.dones = np.delete(self.dones, nrc, axis=0).tolist()
        self.logp_ts = np.delete(self.logp_ts, nrc, axis=0).tolist()



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
        if self.getSize() >= self.max_size:
            self.forget()

        # self.states = np.r_[self.states, [state]]
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.logp_ts.append(logp_t)


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
        self.optimizer = config["optimizer"] # Adam

        # used in getGae()
        self.gamma = config["gamma"]  # 0.99
        self.lamda = config["lambda"]  # 0.90


        max_size = config["memory_size"]  # 100
        forget_ratio = config["forget_ratio"]  # 0.1
        self.memory = ReplayMemory(max_size, forget_ratio)


        self.replay_count = 0

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


    def act(self, state):
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

    def get_gaes(self, rewards, dones, values, next_values, normalize=True):

        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * self.lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)


    def replay(self):

        if not self.is_train :  # no need to replay if it is not the target of training
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

        if USE_TBX:
            self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
            self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
            self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
            self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1


    def loadModel(self, fn):
        self.actor.model.load_weights(f"{fn}_{self.id}_actor.h5")
        self.critic.model.load_weights(f"{fn}_{self.id}_critic.h5")

    def saveModel(self, fn):
        self.actor.model.save_weights(f"{fn}_{self.id}_actor.h5")
        self.critic.model.save_weights(f"{fn}_{self.id}_critic.h5")


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


def makePPOConfig(args):
    config = {}

    config["lr"] = args.lr # 0.005
    config["ppo_epoch"] = args.ppo_epoch # 10
    config["gamma"] = args.gamma  # 0.99
    config["lambda"] = args._lambda # 0.95
    config["actor_lr"] = args.a_lr  # 0.005
    config["critic_lr"] = args.c_lr  # 0.005
    config["ppo_eps"] = args.ppo_eps # 0.1

    config["memory_size"] = args.mem_len
    config["forget_ratio"] = args.mem_fr

    config["offset_range"] = args.offset_range  # 2
    config["control_cycle"] = args.control_cycle # 5
    config["add_time"] = args.add_time # 2

    config["network_layers"] =  [512, 256, 128, 64, 32] # TRAIN_CONFIG['network_size']
    config["optimizer"] = Adam
    return config


if __name__ == "__main__":
    import argparse
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
        args.optimizer = Adam

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
