# Gymnasium Gridworld
# gymnasium_env/envs/grid_world.py
# from https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/


import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MountainCarGridWorldEnv(gym.Env):
    def __init__(self):
        self.size = 16 # The size of the square grid

        self.observation_space = spaces.Discrete(16 * 16) # 256 states in our grid world discretization of MountainCar continuous state space

        self.action_space = gym.spaces.Discrete(3)  # 3 actions: full throttle forward (+1), full throttle reverse (-1), and zero throttle (0)

        # bins for discretization
        # position ranges from -1.2 to 0.6
        # velocity ranges from -0.07 to 0.07
        # # x = bound(-1.2, 0.5)
        self.position_bins = np.linspace(-1.2, 0.6, self.size)  # 15 bins for position
        self.velocity_bins = np.linspace(-0.07, 0.07, self.size)  # 15 bins for velocity

    def discretize(self, state):
        position, velocity = state

        position_idx = np.digitize(position, self.position_bins) - 1
        velocity_idx = np.digitize(velocity, self.velocity_bins) - 1

        # clipping just in case the values are out of bounds
        position_idx = np.clip(position_idx, 0, self.size - 1)
        velocity_idx = np.clip(velocity_idx, 0, self.size - 1)

        return position_idx * self.size + velocity_idx  # return a single integer representing the discrete state, this is the grid that (position, velocity) maps to

    def reset(self, seed=None, options=None):
        # initial values from the book on page 245
        x = np.random.uniform(-0.6, -0.4)
        x_dot = 0
        self.agent_location = (x, x_dot)
        observation = self.discretize(self.agent_location)

        return observation, {} # no info

    def step(self, action):
        position, velocity = self.agent_location

        # apply action
        if action == 0:  # full throttle forward (+1)
            force = 0.001
        elif action == 1:  # full throttle reverse (-1)
            force = -0.001
        else:  # zero throttle (0)
            force = 0.0

        # update velocity and position
        velocity += force - 0.0025 * np.cos(3 * position) # new v = old v + force - gravity pull -> from the book
        velocity = np.clip(velocity, -0.07, 0.07)
        position += velocity
        position = np.clip(position, -1.2, 0.6)

        terminated = False
        # check position for bounds
        if position <= -1.2: # left bound, reset velocity
            velocity = 0.0
        if position >= 0.5: # right bound, terminate episode
            terminated = True

        self.agent_location = (position, velocity)
        # reward structure
        reward = -1.0  # The reward in this problem is -1 on all time steps until the car moves past its goal position at the top of the mountain, which ends the episode

        observation = self.discretize(self.agent_location)

        return observation, reward, terminated, False, {}  # no truncation, no info
