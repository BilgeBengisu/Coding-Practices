import gymnasium as gym
from mountaincar_grid_world import MountainCarGridWorldEnv

gym.register(
    id="MountainCarGridWorld",
    entry_point="mountaincar_grid_world:MountainCarGridWorldEnv",
)
# we are implementing discretization by using grid world like approach
env = gym.make("MountainCarGridWorld")
env.reset()  # reset the environment to start state

# sanity check
print("Action space:", env.action_space)  # Discrete(3)
print("Observation space:", env.observation_space)  # Box([-1.2  -0.07], [0.6  0.07], (2,), float32)

# epsilon-greedy policy
import numpy as np
def epsilon_greedy_policy(state, Q, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # explore
    else:
        return np.argmax(Q[state])  # exploit

# define hyperparameters
num_episodes = 200
alpha = 0.5       # learning rate
gamma = 1.0       # discount factor


Q = np.zeros((env.observation_space.n, env.action_space.n))  # initialize Q-table
# episode loop
for eps in range(num_episodes):
    state, _ = env.reset() # drop info dict
    action = epsilon_greedy_policy(state, Q)
    terminated = False  # Initialize before loop
    while not terminated:
        next_state, reward, terminated, _, _ = env.step(action) # drop truncated and info dict that gets returned
        if terminated:
            Q[state, action] += alpha * (reward + gamma * 0 - Q[state, action])
            break
        next_action = epsilon_greedy_policy(next_state, Q)
        # update Q table with e-sarsa update
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        
        state = next_state
        action = next_action

print("Trained Q-table: ", Q)