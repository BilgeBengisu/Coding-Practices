import gymnasium as gym
import numpy as np
import envs
from e_sarsa import run_e_sarsa
import matplotlib.pyplot as plt

#### Example 6.5 from the textbook: Applying Epsilon-Greedy SARSA to Windy Gridworld
env = gym.make("WindyGridWorld-v0")
env.reset()

# Q table initialization
num_actions = env.action_space.n # .n from gym's discrete objects returns the number of discrete objects in that space
rows = env.observation_space.spaces[0].n
cols = env.observation_space.spaces[1].n  # total number of states in the grid // height * width
Q = np.zeros((rows, cols, num_actions))
# default parameters for e-sarsa are num_episodes=170, alpha=0.5, gamma=1.0, epsilon=0.1
answer_q, time_steps = run_e_sarsa(env, Q)

#print("Learned Q-values:" , answer_q)
print("Time steps per episode:" , time_steps)

#### Plotting like Example 6.5 from the textbook
# plot time steps per episode
plt.plot(time_steps, np.arange(len(time_steps)))
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.title('Time Steps per Episode in Windy Gridworld using Epsilon-Greedy SARSA')
plt.savefig('windy_gridworld_e_sarsa.png')
plt.show()


#### Homework Exercise 6.9 King's Windy Gridworld: Applying Epsilon-Greedy SARSA to Windy Gridworld with 8 (+ 1 staying in place) possible actions
env_king = gym.make("WindyGridWorld-v1")
env_king.reset()

# Q table initialization - renaming the variables with a k_ prefix to avoid confusion
k_num_actions = env_king.action_space.n # .n from gym's discrete objects returns the number of discrete objects in that space
k_rows = env_king.observation_space.spaces[0].n
k_cols = env_king.observation_space.spaces[1].n  # total number of states in the grid // height * width
k_Q = np.zeros((k_rows, k_cols, k_num_actions))
# default parameters for e-sarsa are num_episodes=170, alpha=0.5, gamma=1.0, epsilon=0.1
k_answer_q, k_time_steps = run_e_sarsa(env_king, k_Q)

#print("Learned Q-values:" , answer_q)
print("Time steps per episode:" , k_time_steps)

#### Plotting like Example 6.5 from the textbook
# plot time steps per episode
plt.plot(k_time_steps, np.arange(len(k_time_steps)))
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.title("Time Steps per Episode in Windy Gridworld with King's Moves")
plt.savefig("king_windy_gridworld_e_sarsa.png")
plt.show()