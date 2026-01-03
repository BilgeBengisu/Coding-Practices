import gymnasium as gym
from gymnasium import spaces 
from gymnasium import register
import numpy as np

register(
    id='BlackjackGridWorld-v0',
    entry_point='blackjack_env:BlackjackGridWorldEnv',
)

class BlackjackGridWorldEnv(gym.Env):
    def __init__(self):
        self.height = 20
        self.width = 10
        # 0, 1 == 'hit', 'stick'
        self.action_space = gym.spaces.Discrete(2)
        # observation space is a tuple of available grids (row, column) = (player_sum, dealer_upcard)
        self.observation_space = spaces.Tuple((spaces.Discrete(20), spaces.Discrete(10))) # dealer upcard 1-10 mapped to 0-9 to match Discrete() zero indexing
        # the grid is represented as 0-6 on rows from top to bottom and 0-9 on columns
        # dictionary to represent the moves available
        self.moves = {
            0: 'hit',
            1: 'stick'
        }
        # no goal state = will terminate when player busts or sticks and dealer finishes
        self.current_state = None

    def draw_card(self):
        return np.random.randint(1, 10)  # sample cards uniformly from 1-10 to simplify and make all cards have equal probability

    # helper function to map current state to gridworld representation
    def map_to_grid(self):
        # map the current player sum and dealer upcard to gridworld state representation
        if self.usable_ace == True:
            row = self.player_sum - 2             # rows 0–9 -> sums 2–11
        else:
            row = (self.player_sum - 12) + 10     # rows 10–19 -> sums 12–21

        col = self.dealer_upcard - 1             # columns 0-9 -> dealer upcard 1-10

        return (row, col)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # eliminates the type error for getting an unexpected keyword argument - implement to match gymnasium Env reset signature
        self.ace_count = 0
        self.player_sum = 0
        self.usable_ace = False

        # When the environment restarts it should give the player enough cards to have a total of at least 12 or an ace
        while self.player_sum < 12 and not self.usable_ace:
            card = self.draw_card()
            if card == 1:  # ace
                self.ace_count += 1
                self.player_sum += 11
            else:
                self.player_sum += card
            # adjust for ace if bust
            while self.player_sum > 21 and self.ace_count > 0:
                self.player_sum -= 10
                self.ace_count -= 1
        # update usable ace status
        if self.ace_count > 0:
            self.usable_ace = True 

        # randomly draw dealer upcard
        self.dealer_upcard = self.draw_card()

        self.current_state = self.map_to_grid()
        return self.current_state, {}  # return initial observation and empty info dict ( we could do _get_info here just like in some of the code from class if needed but I chose not to implement info message)
    
    def step(self, action):
        if action == 0:  # hit
            card = self.draw_card()
            if card == 1:  # ace
                self.ace_count += 1
                self.player_sum += 11
            else:
                self.player_sum += card
            # adjust for ace if bust
            while self.player_sum > 21 and self.ace_count > 0:
                self.player_sum -= 10
                self.ace_count -= 1
            # update usable ace status
            if self.ace_count > 0:
                self.usable_ace = True 
            else:
                self.usable_ace = False

            # check for bust
            if self.player_sum > 21:
                terminated = True
                reward = -1  # player busts
                self.current_state = None
            else:
                terminated = False
                reward = 0  # game continues
                # set states - convert to gridworld blackjack state representation
                self.current_state = self.map_to_grid()
        if action == 1:  # stick
            # dealer's turn
            dealer_sum = self.dealer_upcard
            dealer_ace_count = 1 if self.dealer_upcard == 1 else 0
            if dealer_ace_count > 0:
                dealer_sum += 10  # count ace as 11 initially

            while dealer_sum < 17:
                card = self.draw_card()
                if card == 1:
                    dealer_ace_count += 1
                    dealer_sum += 11
                else:
                    dealer_sum += card
                # adjust for ace if bust
                while dealer_sum > 21 and dealer_ace_count > 0:
                    dealer_sum -= 10
                    dealer_ace_count -= 1

            # determine the reward
            if dealer_sum > 21 or self.player_sum > dealer_sum:
                reward = 1  # player wins
            elif self.player_sum < dealer_sum:
                reward = -1  # player loses
            else:
                reward = 0  # draw
            # mark the episode as ended
            terminated = True
            self.current_state = (0, 0) # instead of None so that we don't get NoneType error
        
        truncated = False  # no truncation condition in this env
        return self.current_state, reward, terminated, truncated, {} # this signature agrees with the gymnasium API
