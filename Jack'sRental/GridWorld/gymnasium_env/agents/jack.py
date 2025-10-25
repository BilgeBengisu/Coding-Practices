import numpy as np
MAX_CARS = 20

class Jack:
    def __init__(self, gamma=0.9):
        self._policy = None
        self._value_function = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        self._gamma = gamma
        self._reward = None
        self._step = None
        self._return = None
        self._location = None  # location as (cars_loc1, cars_loc2)

    # def init_jack():
    #     self.policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)

    def step(self, loc, reward):
        self._location = loc
        self._reward = reward
        self._step += 1
        self._gamma_to_n *= self._gamma
        self._return += self._gamma_to_n*reward

    def get_action(self, state):
        cars_loc1, cars_loc2 = state
        return self.policy[cars_loc1, cars_loc2]
    
    def update_state_value(self, state, value):
        cars_loc1, cars_loc2 = state
        self.state_values[cars_loc1, cars_loc2] = value


    # def policy_iteration(self, env):
    
    # def policy_evaluation:
    

    # act greedily
    # def improve_policy(self, env):

