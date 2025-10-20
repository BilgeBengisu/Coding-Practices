# The Gridworld from Chapter 3 of Sutton and Barto
# Abstract GridWorld class for a continuous or episodic task
# David Gillman September 22, 2025
#
# Based on:
# Gymnasium Gridworld
# gymnasium_env/envs/grid_world.py
# from https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

# works when running run_grid_world.py
from ..agents.grid_world_agent import GridWorldAgent

# moved Action Enum to actions directory to avoid circular import - see README
from ..agents.grid_world_actions import GridWorldActions

# Render modes:
#  - human is for animating a sequence of actions in the terminal
#  - rgb_array is for producing an array of pixel values
class GridWorldEnv(gym.Env):
    def __init__(self, render_mode=None, size=5, agent=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.step_num = 0

        self._agent = agent
        self._terminal_locations = []

        # Observations are dictionaries with the agent's location.
        # removed target key
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # Inherited class must fill these in
        self.action_space = None
        self.render_mode = None

        """
        If human-rendering is used, `self.window` will be a reference
        to the window (surface, screen) that we draw to.
        `self.clock` will be a clock that is used to ensure that the
        environment is rendered at the correct framerate in human-mode.
        They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.font = None
        # The size of a single grid square in pixels
        self.pix_square_size = self.window_size / self.size

    def _get_obs(self):
        return {"agent": self._agent.get_location()}

    def _get_info(self):
        return {}

    # default reward is 0
    def _reward(self, curr_loc, step, actual_loc):
        return 0

    # getter, mainly for agent
    def get_terminal_locations(self):
        return self._terminal_locations.copy()

    # options may contain
    # - location, a 2-tuple of coordinates
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location - an ndarray
        initial_loc = self.np_random.integers(0, self.size, size=2, dtype=int)
        if options is not None and "location" in options:
            initial_loc = np.array(options["location"])            
        self._agent.reset(initial_loc)

        observation = self._get_obs()
        info = self._get_info()

        # Inherited class must render frame

        return observation, info

    # returns observation, reward, terminated, False, and info
    # base class only increments step number
    # 
    def step(self, action):
        self.step_num += 1
                
        # Dummy values
        reward = 0
        terminated = False
        observation = self._get_obs()
        info = self._get_info()

        # Inherited class must render frame

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    # Must be implemented by inherited class
    def _render_frame(self):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

            
       
