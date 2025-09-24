# Gymnasium Gridworld
# gymnasium_env/envs/grid_world.py
# from https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


# Render modes:
#  - human is for animating a sequence of actions in the terminal
#  - rgb_array is for producing an array of pixel values
class GridWorldNoTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        #! Observations are dictionaries with the agent's and the target's location.
        # I have (x, y) numpy array observations
        # thus, my observation space should be a Box instead of Discrete
        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int) # no target

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        [row,col] = [y,x]
        """
        self._action_to_direction = {
            Actions.right.value: np.array([0,1]),
            Actions.up.value: np.array([-1,0]),
            Actions.left.value: np.array([0,-1]),
            Actions.down.value: np.array([1,0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # reward
        self.return_so_far = 0
        self.last_reward = 0

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array(
            self.np_random.integers(0, self.size, size=2, dtype=int), dtype=float
        )

        self.return_so_far = 0
        self.last_reward = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._agent_location, {}
    
    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {}

    def step(self, action):
        x, y = self._agent_location
        new_loc = self._agent_location
        reward = 0

        # special states
        # In state A any action takes the agent four steps down, with reward +10.
        # In state B any action takes the agent two steps down, with reward +5.
        if (x, y) == (0, 1):  # State A
            self._agent_location = np.array((4, 1))
            reward = 10
        elif (x, y) == (0, 3):  # State B
            self._agent_location = np.array([2, 3])
            reward = 5
        else:
            # normal moves
            direction = self._action_to_direction[action]
            new_loc = self._agent_location + direction

            # off the grid check
            # If the action would take the agent off the grid, the agent stays put and gets reward -1.
            clipped_loc = np.clip(new_loc, 0, self.size - 1)
            if not np.array_equal(clipped_loc, new_loc):
                reward = -1  # agent tried to move off-grid

            self._agent_location = clipped_loc

        self.last_reward = reward
        self.return_so_far += reward

        # no terminal state
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), reward, terminated, truncated, self._get_info()


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

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # reward flash
        if self.last_reward != 0:
            font = pygame.font.SysFont("Arial", 30)
            reward_text = font.render(f"{self.last_reward:+}", True, (0, 200, 0))
            canvas.blit(reward_text, (10, 10))

        # show expected return
        font = pygame.font.SysFont("Arial", 20)
        return_text = font.render(f"Return: {self.return_so_far}", True, (0, 0, 0))
        canvas.blit(return_text, (10, 40))

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
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
