# 1) imports
import numpy as np
import gymnasium as gym
import time
from math import inf

# 2) create environment
env = gym.make('CartPole-v1')
print(env.action_space.n)

# 3) declare the parameters needed
"""
    - learning rate (alpha) : 0.1
    - discount factor (gamma) : 0.95
    - exploration rate (epsilon) : start at 1, decay over time -> epsilon_decay_value = 0.99995
    - number of episodes
"""
n_actions = env.action_space.n

# 4) set up Q-table - rows = states, columns = actions
q_table = np.zeros((10, n_actions))

# 5) define discrete state
# this will include discreatization: the number of bins used to discretize the continuous state space
# will reflect how coarse or fine the discretization is
NUM_BINS = 10
# state observations
cart_position = [-4.8, 4.8]
cart_velocity = [-inf, inf]
pole_angle = [-0.418, 0.418]
pole_angular_velocity = [-inf, inf]

# 6) Q-Learning Algorithm
# Q-Learning Formula: Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])