# import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

from env.sumo import SUMO_grid_super_v0
from env.salt import SALT_doan_super_v0, SALT_doan_multi_v0

problem = "sumo-grid-super-v0"
# env = SUMO_grid_super_v0()
env = SALT_doan_multi_v0()

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))
num_agents = env.agent_num

upper_bound = env.action_space.high
lower_bound = env.action_space.low

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(self.std_dev) * np.ones(num_actions))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(num_states))
        state_out = layers.Dense(256, activation="relu")(state_input)
        state_out = layers.Dense(256, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(64, activation="relu")(action_input)
        action_out = layers.Dense(64, activation="relu")(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(num_actions)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.

    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

        return [np.squeeze(legal_action)]

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))


# Learning rate for actor-critic models

total_episodes = 10000
# Discount factor for future rewards
gamma = 0.95
# Used to update target networks
tau = 0.005

buffer = []
for i in range(num_agents):
    buffer.append(Buffer(2000, 32))

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):
    print("episode : ", ep)
    prev_states = env.reset()
    episodic_reward = 0

    start = time.time()

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()
        actions = []
        for i in range(len(prev_states)):
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_states[i]), 0)
            _action = buffer[i].policy(tf_prev_state, buffer[i].ou_noise)
            actions.append(_action)

        # Recieve state and reward from environment.
        state, reward, done, info, action_mask = env.step(actions)
        # print(len(state))

        if np.sum(action_mask) > 0:
            action_idx = np.where(action_mask == 1)

            #print(np.round(actions, 2), np.round(np.mean(reward), 3))

            for i in range(len(action_idx)):
                # print(i)
                # print((prev_states[i], actions[i][0], reward[i], state[i]))
                buffer[i].record((prev_states[i], actions[i][0], reward[i], state[i]))
                episodic_reward += reward[i]

            buffer[i].learn()
            # print(buffer)
            buffer[i].update_target(buffer[i].target_actor.variables, buffer[i].actor_model.variables, tau)
            buffer[i].update_target(buffer[i].target_critic.variables, buffer[i].critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    print("episode time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
    avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    if ep % 50 == 0:
        plt.plot(avg_reward_list)
        plt.xlabel("Multi Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.show()

    if ep % 50 == 0:
        for i in range(num_agents):
            buffer[i].actor_model.save_weights("model/multiagent{}_pendulum_actor_epoch_{}.h5".format(i,ep))
            buffer[i].critic_model.save_weights("model/multiagent{}_pendulum_critic_epoch_{}.h5".format(i,ep))
            buffer[i].target_actor.save_weights("model/multiagent{}_pendulum_target_actor.h5".format(i))
            buffer[i].target_critic.save_weights("model/multiagent{}_pendulum_target_critic.h5".format(i))
