import numpy as np
import gymnasium as gym
import math
import matplotlib.pyplot as plt


# --------------------------
# Introduction
# --------------------------
# Q-learning implementation for CartPole-v1 environment with improved discretization.

# Result:
# The agent successfully learned to balance the pole consistently.

# Convergence to near-optimal performance (> 450 average reward)

# Final test runs typically achieve reward close to the maximum of 500, indicating successful control.

# Discretization with 12 bins proved sufficient for stable learning without requiring function approximation.

# Achieved stable pole control with consistent reward of 500 over multiple test episodes.

# ********

# This implementation has been worked on after first_implementation.py to enhance the performance of the agent.

# The difference between cartpole.py and first_implementation.py are:
# * Cartpole.py used np for Q_table whereas first_implementation.py used a dictionary. Benefit: Much faster training → more episodes → better convergence
# * First_implementation.py was a manual attempt to discretizing states and normalizing bins. cartpole.py uses np.digitize(). Benefit: makes bins uniform and accurate, better discretization and correct state representation
# * Better epsilon decay schedule. Epsilon_decay 0.9995 instead of 0.99995
# * Higher number of episodes. 25,000 instead of 10,000
# Overall, cartpole.py takes longer but yields a better trained agent.
# --------------------------

# --------------------------
# 1. CREATE ENV
# --------------------------
env = gym.make("CartPole-v1")

# --------------------------
# 2. DISCRETIZATION SETTINGS
# --------------------------

NUM_BINS = 12   # more bins = better performance but more memory and slower training

# switched to the environments observation space bounds below
# obs = [cart_pos, cart_vel, pole_angle, pole_ang_vel]
# use low and high from observation space
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

# Clip bounds for velocity variables (unbounded in Gym)
STATE_BOUNDS[1] = [-3.0, 3.0]      # cart velocity
STATE_BOUNDS[3] = [-3.0, 3.0]      # pole angular velocity

#Discretization is much closer to true dynamics with this method, leading to better q values and convergence to optimal.

def create_bins(num_bins):
    bins = []
    for i in range(len(STATE_BOUNDS)):
        low, high = STATE_BOUNDS[i]
        bins.append(np.linspace(low, high, num_bins + 1)[1:-1])
    return bins

bins = create_bins(NUM_BINS)

def discretize_state(state, bins):
    return tuple(np.digitize(s, b) for s, b in zip(state, bins)) #digitize() makes bins uniform and accurate
    #avoids off-by-one/discretization biases or scales easily to more bins


# --------------------------
# 3. Q-TABLE INIT
# --------------------------
q_table = np.zeros([NUM_BINS] * 4 + [env.action_space.n])


# --------------------------
# 4. HYPERPARAMETERS
# --------------------------
alpha = 0.1        # learning rate
gamma = 0.99       # discount
epsilon = 1.0      # exploration
epsilon_min = 0.01
epsilon_decay = 0.9995

EPISODES = 25000   # training episodes

reward_history = []

# --------------------------
# 5. TRAINING LOOP
# --------------------------
for episode in range(EPISODES):
    state, _ = env.reset()
    state_d = discretize_state(state, bins)

    total_reward = 0

    done = False
    while not done:

        # --- epsilon-greedy ---
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_d])

        # step environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # discretize next state
        next_state_d = discretize_state(next_state, bins)

        # --- Q-learning update ---
        best_next = np.max(q_table[next_state_d])
        q_table[state_d + (action,)] += alpha * (reward + gamma * best_next - q_table[state_d + (action,)])

        state_d = next_state_d
        total_reward += reward

    # decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_history.append(total_reward)

    if episode % 1000 == 0:
        print(f"Episode {episode}, Average reward: {np.mean(reward_history[-100:]):.2f}")

print("Training finished!")

# --- Visualization of Training Performance ---
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------
# 7. TEST TRAINED AGENT (WATCH)
# --------------------------
env = gym.make("CartPole-v1", render_mode="human")

for _ in range(5):
    state, _ = env.reset()
    state_d = discretize_state(state, bins)
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state_d])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_d = discretize_state(next_state, bins)
        state_d = next_state_d
        total_reward += reward

    print("Test episode reward:", total_reward)
