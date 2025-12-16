import gymnasium as gym
from gymnasium import spaces # http://deepwiki.com/openai/gym/2.3-spaces

# implementing the Windy Gridworld environment
class KingWindyGridWorldEnv(gym.Env):
    def __init__(self):
        self.height = 7
        self.width = 10
        # first the basic action space implementation with 4 actions: up, down, left, right
        self.action_space = gym.spaces.Discrete(9)
        # observation space is a tuple of available grids (row, column)
        self.observation_space = spaces.Tuple(( # pass it as a tuple
            # from documentation:
            # Discrete represents a finite set of possible values, typically integers. This is often used for environments with a fixed number of distinct actions.
            spaces.Discrete(self.height), 
            spaces.Discrete(self.width)
        ))
        # the grid is represented as 0-6 on rows from top to bottom and 0-9 on columns
        # dictionary to represent the moves available
        self.moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (-1, -1), # up-left
            5: (-1, 1),  # up-right
            6: (1, -1),  # down-left
            7: (1, 1),    # down-right
            8: (0,0)     # stay in place
        }
        self.start_state = (3, 0)  # starting position
        self.goal_state = (3, 7)   # goal position
        self.current_state = self.start_state
        # wind strength for each column
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # eliminates the type error for getting an unexpected keyword argument - implement to match gymnasium Env reset signature

        self.current_state = self.start_state
        return self.current_state, {}  # return initial observation and empty info dict ( we could do _get_info here just like in some of the code from class if needed but I chose not to implement info message)
    
    def step(self, action):
        row, col = self.current_state
        # apply wind effect
        wind = self.wind_strength[col]
        row -= wind  # wind pushes the agent up
        
        # apply action effect
        move = self.moves[action]
        row += move[0]
        col += move[1]
        
        # ensure the agent stays within bounds
        row = max(0, min(self.height - 1, row))
        col = max(0, min(self.width - 1, col))
        
        self.current_state = (row, col)
        
        # check if goal is reached
        terminated = self.current_state == self.goal_state
        reward = 1 if terminated else -1  # constant rewqard of -1 until reaching G
        truncated = False  # no truncation condition in this env
        return self.current_state, reward, terminated, truncated, {} # info dict is empty
        # step function in openAI Gym envs returs a tuple of (observation, reward, done, info) we return empty for that info dict
