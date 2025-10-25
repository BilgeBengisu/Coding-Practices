"""
Agent class for GridWorld
David Gillman
September 21, 2025
"""

import numpy as np
from sys import float_info
from .grid_world_policy import GridWorldPolicy
from .grid_world_actions import GridWorldActions

class GridWorldAgent():
    def __init__(self, size, gamma=0.9):
        self._size = size
        self._location = None
        self._reward = None
        self._step = None
        self._return = None
        self._gamma = gamma
        self._gamma_to_n = None
        self._policy = np.empty((size, size), dtype=object) # a dictionary
        self._value_function = np.zeros((size, size), dtype=float)

        # random initial policy
        self.set_policy_random()
        
    def _policy2string(self):
        action_char = {GridWorldActions.right:"r",
                       GridWorldActions.up:"u",
                       GridWorldActions.left:"l",
                       GridWorldActions.down:"d"
                      }
        strpol = np.empty((self._size, self._size), dtype=np.dtype('U4') ) 
        for i in range(self._size):
            for j in range(self._size):
                s = ""
                for a in action_char:
                    if a in self._policy.T[i, j]:
                        s += action_char[a]
                strpol[i, j] = s
        return strpol

    # exposing this for experiments
    def set_policy_random(self):        
        rp = {GridWorldActions.right:0.25,
              GridWorldActions.up:0.25,
              GridWorldActions.left:0.25,
              GridWorldActions.down:0.25
             }
        for i in range(self._size):
            for j in range(self._size):
                self._policy[i, j] = rp.copy()

    # sets initial values
    def reset(self, location):
        self._location = location
        self._reward = None
        self._step = 0
        self._return = 0
        self._gamma_to_n = 1

    # sets the new location and reward
    # Arguments:
    #  - location, a length-2 ndarray of integers
    #  - reward, a number
    def step(self, loc, reward):
        self._location = loc
        self._reward = reward
        self._step += 1
        self._gamma_to_n *= self._gamma
        self._return += self._gamma_to_n*reward

    def get_action(self):
        x = np.random.random()
        thresh = 0.0
        for action, prob in self._policy[tuple(self._location)].items():
            thresh += prob
            if x <= thresh:
                return action.value
        assert(False) # probabilities in a policy should sum to 1
        
    # sets the location as a length-2 ndarray of integers
    def set_location(self, loc):
        self._location = loc
        
    # gets the location 
    def get_location(self):
        return self._location

    # sets the reward as a number
    def set_reward(self, rew):
        self._reward = rew
        
    # gets the reward
    def get_reward(self):
        return self._reward

    # sets the return as a number
    def set_return(self, rew):
        self._return = rew
        
    # gets the return
    def get_return(self):
        return self._return

    # sets the policy for a state
    # Arguments:
    #  ij - a 2-tuple of (graphics) coordinates
    #  d - a dictionary of action-probabilities
    def set_policy(self, ij, d):
        assert(ij[0] < self._size and ij[1] < self._size)
        self._policy[ij[0], ij[1]] = d.copy()

    def get_policy(self):
        return self._policy.copy()

    def get_policy_pretty(self):        
        return np.array2string(self._policy2string())

    def set_value_function(self, arr):
        temp = np.array(arr)
        assert(self._value_function.shape == temp.shape)
        self._value_function = temp

    def get_value_function_pretty(self):
        np.set_printoptions(suppress=True)
        return np.array2string(self._value_function.T, precision=2)

    # performs one step of policy evaluation
    # uses for loop to do this in place
    def evaluate_policy_once(self, env):
        eps = 0.0
        for j in range(self._size):
            for i in range(self._size):
                if (i, j) in env.get_terminal_locations():
                    assert(self._value_function[i, j] == 0.0)
                    continue
                new_value = 0.0
                for action, prob in self._policy[i, j].items():
                    env.reset(options = {"location": (i, j)})
                    env.step(action.value)
                    next_state_value = self._value_function[tuple(self._location)]
                    new_value += prob*(self._reward +
                                       self._gamma*next_state_value)
                eps = max(eps, abs(self._value_function[i, j] - new_value))
                self._value_function[i, j] = new_value
        return eps

    # performs iterative policy evaluation to a given tolerance
    # uses for loop to do this in place
    def evaluate_policy(self, env, tol):
        step = 0
        eps = tol + 1
        while eps > tol:
            eps = self.evaluate_policy_once(env)
            step += 1
        return step

    # updates the policy to greedy for the current value function
    # uses for loop to do this in place
    # Arguments:
    #    env - the environment
    # 
    def improve_policy_once(self, env, debug=False):
        policy_stable = True
        for j in range(self._size):
            for i in range(self._size):
                if (i, j) in env.get_terminal_locations():
                    continue
                new_actions = []
                best_value = -float_info.max
                #for action, prob in self._policy[i, j].items():
                for action in GridWorldActions:
                    env.reset(options = {"location": (i, j)})
                    env.step(action.value)
                    next_state_value = self._value_function[tuple(self._location)]
                    # fix bug
                    if debug and (i, j) == (1, 3):
                        print(action, next_state_value)
                    if next_state_value == best_value:
                        new_actions.append(action)
                    elif next_state_value > best_value:
                        best_value = next_state_value
                        new_actions = [action]
                assert len(new_actions) > 0
                prob = 1./len(new_actions)
                new_policy = dict()
                for action in new_actions:
                    new_policy[action] = prob
                if self._policy[i, j] != new_policy:
                    self._policy[i, j] = new_policy
                    policy_stable = False
        return policy_stable

    # performs one step of policy iteration
    # uses for loop to do this in place
    # Arguments:
    #    env - the environment
    #    tol - tolerance for policy evaluation
    # 
    def iterate_policy_once(self, env, tol):
        steps = self.evaluate_policy(env, tol)
        return self.improve_policy_once(env)

    # performs policy iteration to a given tolerance
    # uses for loop to do this in place
    #! stubbed
    def iterate_policy(self, env, tol):
        step = 0
        while True:
            policy_stable = self.iterate_policy_once(env, tol)
            step += 1
            if policy_stable:
                break
        return step

    
