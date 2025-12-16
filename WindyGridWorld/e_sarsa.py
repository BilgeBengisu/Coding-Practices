import numpy as np

def run_e_sarsa(env, Q, num_episodes=170, alpha=0.5, gamma=1.0, epsilon=0.1):
    ######
    # h elper function to executer epsilon greedy policy
    def epsilon_greedy_policy(Q, state, epsilon, n_actions):
        row, col = state
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)  # explore
        else:
            return np.argmax(Q[row, col, :])  # exploit 
        
    ####

    time_steps = np.zeros(num_episodes)  # to keep track of time steps per episode
    t = 0 # accumulates over episodes
    ## run e-greedy sarsa algorithm
    for n in range(num_episodes):
        state, _ = env.reset() # returns observation and info (info is an empty dict, drop it)
        action = epsilon_greedy_policy(Q, state, epsilon, env.action_space.n)
        
        terminated = False
        while not terminated:
            next_state, reward, terminated, _, _ = env.step(action) # drop truncated and info dict that gets returned
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.action_space.n)
            row, col = state
            nrow, ncol = next_state
            # update Q table with epsilon greedy sarsa update
            Q[row, col, action] += alpha * (reward + gamma * Q[nrow, ncol, next_action] - Q[row, col, action])
            
            state = next_state
            action = next_action
            t += 1
        time_steps[n] = t  # record time steps for this episode

    return Q, time_steps