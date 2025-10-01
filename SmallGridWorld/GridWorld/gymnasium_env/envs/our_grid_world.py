# The Gridworld from Chapter 3 of Sutton and Barto
# A continuous task
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

MAX_NUM_STEPS = 20

# Render modes:
#  - human is for animating a sequence of actions in the terminal
#  - rgb_array is for producing an array of pixel values
class OurGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 2, "steps": MAX_NUM_STEPS,
                "gamma": 0.9}

    def __init__(self, render_mode=None, size=5, agent=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.step_num = 0

        self._agent = agent

        # Observations are dictionaries with the agent's location.
        # removed target key
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the step we will take if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._default_action_to_step = {
            GridWorldActions.right.value: np.array([1, 0]),
            GridWorldActions.up.value: np.array([0, -1]),
            GridWorldActions.left.value: np.array([-1, 0]),
            GridWorldActions.down.value: np.array([0, 1]),
        }
        self._location_action_to_step = {
            (1,0): {
                GridWorldActions.right.value: np.array([0, 4]),
                GridWorldActions.up.value: np.array([0,4]),
                GridWorldActions.left.value: np.array([0, 4]),
                GridWorldActions.down.value: np.array([0, 4]),
            },
            (3,0): {
                GridWorldActions.right.value: np.array([0, 2]),
                GridWorldActions.up.value: np.array([0, 2]),
                GridWorldActions.left.value: np.array([0, 2]),
                GridWorldActions.down.value: np.array([0, 2]),
            }
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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
        # removed target key
        return {"agent": self._agent.get_location()}

    def _get_info(self):
        # removed distance from target
        return {}

    # default reward is 0
    # -1 for stepping off grid
    # two locations get special rewards
    def _reward(self, curr_loc, step, actual_loc):
        loc = tuple(curr_loc)
        if loc == (1, 0):
            return 10
        elif loc == (3, 0):
            return 5
        elif not np.array_equal(actual_loc, curr_loc + step): # stepped outside grid
            return -1
        else:
            return 0

    # options may contain
    # - location, a 2-tuple of coordinates
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location - an ndarray
        initial_loc = self.np_random.integers(0, self.size, size=2, dtype=int)
        if options is not None and "location" in options:
            initial_loc = np.array(options["location"])            
        self._agent.reset(initial_loc, self.metadata["gamma"])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # returns observation (agent location), reward, terminated (False), False, and info
    # has the side effects of
    # - setting our agent's location
    # - setting our agent's reward
    # 
    def step(self, action):
        self.step_num += 1
        # Get the action-to-step dictionary from the current agent location
        curr_location = self._agent.get_location()
        default_a2s = self._default_action_to_step
        loc = tuple(curr_location)
        this_a2s = self._location_action_to_step.get(loc, default_a2s)

        # Map the action (0, 1, 2, 3) to the step we take
        this_step = this_a2s[action]

        # the step may take the agent off the grid; clip the next location to within the grid
        next_location = curr_location + this_step 
        real_next_location = np.clip(next_location, 0, self.size - 1)
        
        # The task is continuing
        terminated = False if self.step_num <= self.metadata["steps"] else True
        reward = self._reward(curr_location, this_step, real_next_location)
        self._agent.step(real_next_location, reward)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, self._agent.get_reward(), terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.font is None and self.render_mode == "human":
            self.font = pygame.font.Font(None, int(self.pix_square_size / 4))

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Now we draw the agent
        assert(self._agent.get_location() is not None)
        agent_center = (self._agent.get_location() + 0.5) * self.pix_square_size
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            agent_center,
            self.pix_square_size / 3,
        )

        # agent reward and return - to be blitted on top of canvas later
        offset = np.array((0, self.pix_square_size / 6))
        rwd_img = self.font.render(str(self._agent.get_reward()), True,
                                   (255, 255, 255))
        rwd_rect = rwd_img.get_rect()        
        rwd_rect.center = agent_center - offset
        rtn_img = self.font.render(f"{self._agent.get_return():.2f}", True,
                                   (255, 255, 255))
        rtn_rect = rtn_img.get_rect()        
        rtn_rect.center = agent_center + offset

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas`
            # to the visible window
            self.window.blit(canvas, canvas.get_rect())
            self.window.blit(rwd_img, rwd_rect)
            self.window.blit(rtn_img, rtn_rect)
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the
            # predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

## New methods for OurGridWorldEnv
            
       
