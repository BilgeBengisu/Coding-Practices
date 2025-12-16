import gymnasium as gym
import numpy as np
from windy_gridworld_env import WindyGridWorldEnv
from e_sarsa import run_e_sarsa
import matplotlib.pyplot as plt

env = gym.make("WindyGridWorld-v0")
env.reset()

# Q table initialization
num_actions = env.action_space.n # .n from gym's discrete objects returns the number of discrete objects in that space
rows = env.observation_space.spaces[0].n
cols = env.observation_space.spaces[1].n  # total number of states in the grid // height * width
Q = np.zeros((rows, cols, num_actions))
# default parameters for e-sarsa are num_episodes=170, alpha=0.5, gamma=1.0, epsilon=0.1
answer_q, time_steps = run_e_sarsa(env, Q)

print("Learned Q-values:" , answer_q)
print("Time steps per episode:" , time_steps)

#### Plotting like Example 6.5 from the textbook
# plot time steps per episode
plt.plot(time_steps, np.arange(len(time_steps)))
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.title('Time Steps per Episode in Windy Gridworld using Epsilon-Greedy SARSA')
plt.show()