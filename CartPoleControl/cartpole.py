# 1) imports
import numpy as np
import gymnasium as gym
import time
from math import inf

# 2) create environment
env = gym.make('CartPole-v1', render_mode='human')
print(env.action_space.n)

# 3) declare the parameters needed
"""
    - learning rate (alpha) : 0.1
    - discount factor (gamma) : 0.95
    - exploration rate (epsilon) : start at 1, decay over time -> epsilon_decay_value = 0.99995
    - number of episodes
"""
n_actions = env.action_space.n

########### Sanity Check ############
# obs, info = env.reset()
# done = False
# for _ in range(200):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated: # check if episode is over
#         obs, info = env.reset()
# env.close()
#####################################

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

def discretize_state(state):
    return

# 6) Q-Learning Algorithm
# Q-Learning Formula: Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

# epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore action space
    else:
        return np.argmax(q_table[state])  # Exploit learned values
    

# run episodes
n_episodes = 10000
for episode in range(n_episodes):
    continue



env.close()