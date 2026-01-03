"""
First-Visit Monte Carlo Policy Evaluation for Gridworld Blackjack Environment
Hints:
Implement the environment as a 2D GridWorld. One dimension is the 10 possible up cards of the dealer. 
The other dimension has 20 values: 2-11 represent the player's total when the player has a usable ace, 
and 12-21 represent the player's total when the player has no usable ace. 
When the environment restarts it should give the player enough cards to have a total of at least 12 or an ace.
"""

## create environment as a 2d gridworld 
# rewards +1 for winning, -1 for losing, 0 for draw
# 20 rows (y-axis: player states) x 10 columns (x-axis: dealer states) = 200 states/grids
# rows 0–9 -> player sum 2–11 with usable ace(soft hands), rows 10–19 -> player sum 12–21 with no usable ace (hard hands) -- this design help us only focus on the meaningful states
# states = (player_total, dealer_upcard)
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from blackjack_env import BlackjackGridWorldEnv
from collections import defaultdict
from matplotlib import pyplot as plt

env = gym.make('BlackjackGridWorld-v0')
env.reset()


# define hyperparameters

# initialize 

# fixed policy: the policy that sticks if the player’s sum is 20 or 21, and otherwise hits
def policy(state):
    row, _ = state
    player_sum = row + 2 # map back to player sum -> similar to map_to_grid() function in env

    if player_sum >= 20:
        return 1  # stick
    else:
        return 0  # hit

# generate episode
def generate_episode(env):
    episode = []
    state = env.reset()[0]  # reset environment to start a new episode
    terminal = False
    while terminal == False:
        # take action according to epsilon-greedy policy
        action = policy(state)
        # take a step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action) # drop the info dict
        # append the step to the episode to track the trajectory
        episode.append((state, action, reward))
        # move to next state
        state = next_state 
        # update terminal to check if episode ended
        terminal = terminated

    return episode

# first-visit MC prediction
def first_visit_mc(env, num_episodes, gamma = 1.0):
    Q = defaultdict(float)  # initialize action-value function as defaultdict to avoid key error
    returns = defaultdict(list)
    #mc_state_values = np.zeros(env.observation_space.n)  # initialize state value function
    for _ in range(num_episodes):
        # take actions according to epsilon-greedy policy
        episode = generate_episode(env)
        visited = set()
        for t , (state, action, _) in enumerate(episode): # itirate backwards to calculate G as the sum of rewards from t to the end
            if (state, action) in visited:
                continue  # skip if already visited
            visited.add((state, action)) # make sure to mark as visited
            G = 0
            for k, (_, _, reward) in enumerate(episode[t:]): # don't need state, and action here
                G += (gamma ** k) * reward # accumalted G from time step onwards

            returns[(state, action)].append(G)
            Q[(state, action)] = np.mean(returns[(state, action)]) # update action-value function as average of returns
    return Q

# implementing first-visit monte carlo for 10,000 episodes
num_eps = 10000
q_from_10000_eps = first_visit_mc(env, num_eps)
#print(q_from_10000_eps)
# using q_from_10000_eps, we can extract the state-value function by taking the max over actions for each state
V = defaultdict(float)
for (state, action), value in q_from_10000_eps.items():
    if state not in V or value > V[state]:
        V[state] = value
print(V)

# replicate Figure 5.1 as heatmap
heatmap = np.zeros((20, 10))
heatmap = np.full((20, 10), np.nan)

for (row, dealer), value in V.items():
    heatmap[row, dealer] = value

plt.figure(figsize=(10, 6))
plt.imshow(heatmap, origin="lower", cmap="coolwarm")
plt.colorbar(label="State Value V(s)")

plt.xlabel("Dealer Upcard")
plt.ylabel("Player State (row index)")

plt.title("First-Visit Monte Carlo State Values")

plt.show()
plt.savefig("first_visit_mc_blackjack_values.png")