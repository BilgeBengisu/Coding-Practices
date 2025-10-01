"""
Agent class for GridWorld
David Gillman
September 21, 2025
"""

import numpy as np
from .grid_world_policy import GridWorldPolicy
from .grid_world_actions import GridWorldActions

class GridWorldAgent():
    def __init__(self, size):
        self._size = size
        self._location = None
        self._reward = None
        self._step = None
        self._return = None
        self._gamma = None # from environment
        self._gamma_to_n = None
        self._policy = np.empty((size, size), dtype=object) # a dictionary
        self._value_function = np.zeros((size, size), dtype=float)

        # random initial policy
        rp = {GridWorldActions.right:0.25,
              GridWorldActions.up:0.25,
              GridWorldActions.left:0.25,
              GridWorldActions.down:0.25
             }
        for i in range(size):
            for j in range(size):
                self._policy[i, j] = rp.copy()

    # sets initial values
    def reset(self, location, gamma):
        self._location = location
        self._reward = None
        self._step = 0
        self._return = 0
        self._gamma = gamma
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

    # performs one step of policy iteration
    # uses for loop to do this in place
    #! stubbed
    def iterate_policy_once(self, env):
        policy_stable = True
        for j in range(self._size):
            for i in range(self._size):
                old_action = max(self._policy[i, j], key=self._policy[i, j].get)

                # computing Q(s,a) for all actions
                action_values = {}
                for action in GridWorldActions:
                    env.reset(options={"location": (i, j)})
                    env.step(action.value)
                    next_state_value = self._value_function[tuple(self._location)]
                    action_values[action] = self._reward + self._gamma * next_state_value

                # greedy action
                best_action = max(action_values, key=action_values.get)

                # update policy to be deterministic
                new_policy = {a: 0.0 for a in GridWorldActions}
                new_policy[best_action] = 1.0
                self._policy[i, j] = new_policy

                # check stability
                if best_action != old_action:
                    policy_stable = False

        return policy_stable

    # performs policy iteration to a given tolerance
    # uses for loop to do this in place
    #! stubbed
    def iterate_policy(self, env, tol):
        step = 0
        while True:
            # evaluating policy
            self.evaluate_policy(env, tol)

            # improving policy if not stable
            policy_stable = self.iterate_policy_once(env)
            step += 1

            if policy_stable:
                break
        return step

    
