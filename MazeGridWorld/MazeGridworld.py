import numpy as np

class MazeGridworld:
    def __init__(self, height, width, start, goal, walls):
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.walls = set(walls)
        
        # Actions: up, down, left, right
        self.actions = {
            0: (-1, 0),   # up
            1: (1, 0),    # down
            2: (0, -1),   # left
            3: (0, 1)     # right
        }

    def step(self, state, action):
        if state == self.goal:
            return state, 0   # goal terminal, no movement

        dr, dc = self.actions[action]
        nr, nc = state[0] + dr, state[1] + dc
        
        # Out of bounds → stay
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return state, -1
        
        # Wall → stay
        if (nr, nc) in self.walls:
            return state, -1
        
        # Normal move
        new_state = (nr, nc)
        reward = 0 if new_state == self.goal else -1
        return new_state, reward

    def reset(self):
        return self.start
