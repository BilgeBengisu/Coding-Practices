'''
Optimal policy of the GridWorldAgent for OurGridWorldEnv.
'''

import numpy as np
from gymnasium_env.agents.grid_world_actions import GridWorldActions

SIZE = 5
OPTIMAL_POLICY = np.empty((SIZE, SIZE), dtype=object)
OPTIMAL_POLICY[0, 0] = {GridWorldActions.right: 1.0}
for i, j in [(1, 0), (3, 0)]:
    OPTIMAL_POLICY[i, j] = {GridWorldActions.right:0.25,
                            GridWorldActions.up:0.25,
                            GridWorldActions.left:0.25,
                            GridWorldActions.down:0.25
                           }
for i, j in [(2, 0), (4, 0), (3, 1), (4, 1)]:
    OPTIMAL_POLICY[i, j] = {GridWorldActions.left:1.0}
for i, j in [(0, 1), (0, 2), (0, 3), (0, 4)]:
    OPTIMAL_POLICY[i, j] = {GridWorldActions.right:0.5,
                            GridWorldActions.up:0.5
                           }
for i, j in [(1, 1), (1, 2), (1, 3), (1, 4)]:
    OPTIMAL_POLICY[i, j] = {GridWorldActions.up:1.0}
for i, j in [(2, 1), (2, 2), (2, 3), (2, 4),
                     (3, 2), (3, 3), (3, 4),
                     (4, 2), (4, 3), (4, 4)]:
    OPTIMAL_POLICY[i, j] = {GridWorldActions.left:0.5,
                            GridWorldActions.up:0.5
                           }
