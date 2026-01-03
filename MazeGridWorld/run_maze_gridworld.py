import MazeGridworld
import random
import numpy as np
import matplotlib.pyplot as plt

# create the maze from the book using our MazeGridworld class
OurMaze = MazeGridworld.MazeGridworld(
    height=5,
    width=6,
    start=(0,0),
    goal=(4,5),
    walls={(0,2),(1,2),(2,2),(3,2),(3,5)}
)

# helper
def state_to_index(state, width):
    return state[0]*width + state[1]
# helper to plot the policy later
def extract_greedy_policy(Q):
    return np.argmax(Q, axis=1)

# ε-greedy action selection
def epsilon_greedy(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[s])

# dyna-q
def dyna_q(env, n_planning, episodes=50, alpha=0.1, gamma=0.95, epsilon=0.01):
    H, W = env.height, env.width
    n_states = H*W
    n_actions = 4 # up down left right
    
    Q = np.zeros((n_states, n_actions))
    model = {}  # dictionary model
    
    steps_per_episode = []
    state_visits_ep1 = np.zeros(n_states, dtype=int)
    Q_end_of_episode = []
    Q_episode2_steps = []
    
    for ep in range(episodes):
        s = env.reset()
        s_idx = state_to_index(s, W)

        steps = 0
        
        while True:
            # Logging: second episode Q after each step
            if ep == 1:
                Q_episode2_steps.append(Q.copy())

            # Count visits in episode 1
            if ep == 0:
                state_visits_ep1[s_idx] += 1

            # ε-greedy action
            a = epsilon_greedy(Q, s_idx, epsilon)
            
            # Real environment step
            s2, r = env.step(s, a)
            s2_idx = state_to_index(s2, W)

            # Q-learning update
            Q[s_idx, a] += alpha * (r + gamma * np.max(Q[s2_idx]) - Q[s_idx, a])
            
            # Insert into model
            model[(s_idx, a)] = (s2_idx, r)

            # Planning
            for _ in range(n_planning):
                # Pick a random past state-action
                (sp, ap), (sp2, rp) = random.choice(list(model.items()))
                Q[sp, ap] += alpha * (rp + gamma * np.max(Q[sp2]) - Q[sp, ap])
            
            steps += 1
            s, s_idx = s2, s2_idx

            if s == env.goal:
                break

        steps_per_episode.append(steps)
        Q_end_of_episode.append(Q.copy())

    # return {
    #     "steps_per_episode": steps_per_episode,
    #     "state_visits_ep1": state_visits_ep1,
    #     "Q_end_of_episode": Q_end_of_episode,
    #     "Q_episode2_steps": Q_episode2_steps
    # }

    return steps_per_episode, Q, state_visits_ep1


#######################################
# Run Dyna-Q
steps0, Q0, visits0 = dyna_q(OurMaze, n_planning=0)
steps50, Q50, visits50 = dyna_q(OurMaze, n_planning=50)


# -------------------------------
# Plotting (like Figure 8.2)
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps0, label="0 planning steps (direct RL)")
plt.plot(steps50, label="50 planning steps")
plt.xlabel("Episodes")
plt.ylabel("Steps per episode")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# Simpler approach to plotting Figure 8.3 - providing the information without graphics
# -------------------------------
policy0 = extract_greedy_policy(Q0)
policy50 = extract_greedy_policy(Q50)

print("Greedy policy for n=0:\n", policy0)
print("\nGreedy policy for n=50:\n", policy50)

print("\nState visit counts in episode 1 (n=0):\n", visits0)
print("\nState visit counts in episode 1 (n=50):\n", visits50)
