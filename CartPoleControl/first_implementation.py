# 1) imports
import numpy as np
import gymnasium as gym
import time
from math import inf

# 2) create environment
env = gym.make('CartPole-v1', render_mode='rgb_array') # rgb_array for training purposes
print(env.action_space.n)

# 3) declare the parameters needed
"""
    - learning rate (alpha) : 0.1
    - discount factor (gamma) : 0.95
    - exploration rate (epsilon) : start at 1, decay over time 
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

# 4) define discrete state
# this will include discreatization: the number of bins used to discretize the continuous state space
# will reflect how coarse or fine the discretization is
NUM_BINS = 10
# state observations bounds
cart_position_bounds = [-4.8, 4.8]
cart_velocity_bounds = [-5.0, 5.0] 
pole_angle_bounds = [-0.418, 0.418]
pole_angular_velocity_bounds = [-5.0, 5.0]

def discretize_state(state):
    """
    Convert continuous state to discrete state index.
    State has 4 continuous values: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Returns a tuple of 4 discrete indices.
    """
    discretized = []
    bounds = [cart_position_bounds, cart_velocity_bounds, pole_angle_bounds, pole_angular_velocity_bounds]
    
    for i, (value, bound) in enumerate(zip(state, bounds)):
        # Clip value to bounds
        value = np.clip(value, bound[0], bound[1])
        # Normalize to [0, 1]
        normalized = (value - bound[0]) / (bound[1] - bound[0])
        # Discretize to bin index
        bin_index = int(normalized * (NUM_BINS - 1))
        discretized.append(bin_index)
    
    return tuple(discretized)

# 5) set up Q-table
# Q-table shape: (NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, n_actions)
# One dimension for each state variable, plus one for actions
q_table = {}  # Use dictionary for sparse representation

# 6) Q-Learning Algorithm
# Q-Learning Formula: Q[state, action] += α * (reward + γ * max(Q[next_state]) - Q[state, action])

# hyperparameters
alpha = 0.1  # learning rate
gamma = 0.95  # discount factor
epsilon = 1.0  # exploration rate
epsilon_decay = 0.99995  # epsilon decay rate

# epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore action space
    else:
        # Return action with highest Q-value for this state
        if state in q_table:
            return np.argmax(q_table[state])
        else:
            return env.action_space.sample()

# run episodes
n_episodes = 10000
episode_rewards = []
episode_lengths = []

for episode in range(n_episodes):
    obs, info = env.reset()
    current_state = discretize_state(obs)
    
    # Initialize Q-values for this state if not seen before
    if current_state not in q_table:
        q_table[current_state] = np.zeros(n_actions)
    
    episode_reward = 0
    episode_length = 0
    done = False
    
    while not done:
        # Choose action using epsilon-greedy policy
        action = epsilon_greedy_policy(current_state, epsilon)
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(obs)
        done = terminated or truncated
        
        # Initialize Q-values for next state if not seen before
        if next_state not in q_table:
            q_table[next_state] = np.zeros(n_actions)
        
        # Get max Q-value for next state
        max_next_q = np.max(q_table[next_state])
        
        # Q-Learning update
        # Q[state, action] += α * (reward + γ * max(Q[next_state]) - Q[state, action])
        q_table[current_state][action] += alpha * (reward + gamma * max_next_q - q_table[current_state][action])
        
        episode_reward += reward
        episode_length += 1
        current_state = next_state
    
    # Decay epsilon
    epsilon *= epsilon_decay
    
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    # Print progress
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(episode_rewards[-500:])
        avg_length = np.mean(episode_lengths[-500:])
        print(f"Episode {episode + 1}/{n_episodes}, Avg Reward (last 500): {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Epsilon: {epsilon:.5f}")

print(f"\nTraining complete!")
print(f"Total states visited: {len(q_table)}")
print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")

env.close()


# Watch trained agent in human render mode for multiple episodes
n_human_episodes = 5
env_human = gym.make("CartPole-v1", render_mode="human")
seconds = 0.05

for episode in range(1, n_human_episodes + 1):
    obs, info = env_human.reset()
    total_reward = 0
    steps = 0
    print(f"\nEpisode {episode}:")

    while True:
        state = discretize_state(obs)
        action = np.argmax(q_table[state]) if state in q_table else env_human.action_space.sample()
        obs, reward, terminated, truncated, info = env_human.step(action)
        total_reward += reward
        steps += 1

        if steps % 5 == 0:
            print(f"Step {steps}, Reward: {reward}")
        time.sleep(seconds) # to slow down the rendering

        if terminated or truncated:
            print(f"Episode {episode} finished after {steps} steps, total reward: {total_reward}")
            break

env_human.close()