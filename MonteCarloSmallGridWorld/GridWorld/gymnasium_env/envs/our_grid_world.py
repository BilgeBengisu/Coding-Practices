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
from .grid_world import GridWorldEnv

# works when running run_grid_world.py
from ..agents.grid_world_agent import GridWorldAgent

# moved Action Enum to actions directory to avoid circular import - see README
from ..agents.grid_world_actions import GridWorldActions

MAX_NUM_STEPS = 20

# Render modes:
#  - human is for animating a sequence of actions in the terminal
#  - rgb_array is for producing an array of pixel values
class OurGridWorldEnv(GridWorldEnv):
    metadata = {"render_modes": ["human", "rgb_array"],
                "render_fps": 2, "steps": MAX_NUM_STEPS}

    def __init__(self, render_mode=None, size=5, agent=None):
        super().__init__(render_mode=render_mode, size=size, agent=agent)

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
        observation, info = super().reset(seed=seed, options=options)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # returns observation (agent location), reward, terminated (False), False, and info
    # has the side effects of
    # - setting our agent's location
    # - setting our agent's reward
    # 
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

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
            
       
