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
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        #! Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([0, 1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.down.value: np.array([1, 0]),
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

        # Running total reward for the current episode. Use
        # `get_total_reward()` and `clear_total_reward()` to query/clear.
        self._total_reward = 0

    def _get_obs(self):
        #!
        return {"agent": self._agent_location}

    def _get_info(self):
        #!
        return {
            "agent location": tuple(int(x) for x in self._agent_location),
            "total_reward": float(self._total_reward),
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        #! We will sample the target's location randomly until it does not
        # coincide with the agent's location
        # """ self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     ) """

        # Reset running total reward at the start of an episode
        self._last_reward = 0
        self._total_reward = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # Compute tentative new location and clip to grid boundaries.
        new_loc = self._agent_location + direction

        clipped_loc = np.clip(new_loc, 0, self.size - 1)
        # Off-grid movement penalty takes precedence: if the agent tried
        # to step out of bounds we penalize and place it on the border.
        if not np.array_equal(clipped_loc, new_loc):
            reward = -1  # agent tried to move off-grid
            self._agent_location = clipped_loc
        else:
            # Valid move; update agent location and check for special states
            self._agent_location = new_loc
            reward = 0
            # Handle special teleport/reward states (A and B)
            reward += self._handle_special_states()

        #! An episode is done iff the agent has reached the target
        terminated = False  # np.array_equal(self._agent_location, self._target_location)

        self._last_reward = reward

        # Update running total and build return values
        self._total_reward += reward

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _handle_special_states(self):
        """Check for special states and apply teleport + reward.

        Returns the reward granted by the special state (0 if none).
        State A: at (0,1) -> teleport to (4,1), reward 10
        State B: at (0,3) -> teleport to (2,3), reward 5
        """
        # (0,1) and (0,3) are the special states
        if np.array_equal(self._agent_location, np.array([1,0])):
            # Teleport to (4,1)
            self._agent_location = np.array([1, 4], dtype=int)
            return 10
        if np.array_equal(self._agent_location, np.array([3,0])):
            # Teleport to (2,3)
            self._agent_location = np.array([3, 2], dtype=int)
            return 5
        return 0

    def get_total_reward(self):
        """Return the running total reward for the current episode."""
        return self._total_reward

    def clear_total_reward(self, reset_to: float = 0.0):
        """Clear (or set) the running total reward dynamically.

        Useful when you want to read and then reset the total reward at
        arbitrary points during training/evaluation.
        """
        self._total_reward = reset_to

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

        #! First we draw the target
        # """ pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # ) """
        #! Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # reward flash
        if self._last_reward != 0:
            font = pygame.font.SysFont("Arial", 30)
            reward_text = font.render(f"{self._last_reward:+}", True, (0, 200, 0))
            canvas.blit(reward_text, (10, 10))

        # show expected return
        font = pygame.font.SysFont("Arial", 20)
        return_text = font.render(f"Return: {self._total_reward}", True, (0, 0, 0))
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
# The file above accidentally contained a duplicated copy; the next update
# removes the duplicated content and keeps the single corrected class.
