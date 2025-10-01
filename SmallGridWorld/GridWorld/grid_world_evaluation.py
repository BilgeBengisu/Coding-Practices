# Run `pip install "gymnasium[classic-control]"` for this example.
import numpy as np
import gymnasium as gym
import gymnasium_env
from optimal_policy import OPTIMAL_POLICY
from gymnasium_env.agents.grid_world_agent import GridWorldAgent
from gymnasium_env.agents.grid_world_actions import GridWorldActions

agent = GridWorldAgent(5)
env = gym.make("gymnasium_env/OurGridWorld-v0", render_mode=None, agent=agent)
observation, info = env.reset()

tol = 0.01

print("random policy...")
agent.evaluate_policy(env, tol)
print("value functions under random policy:")
print(agent.get_value_function_pretty())

print("running policy iteration once")
steps = agent.iterate_policy_once(env)

print(agent.get_value_function_pretty())

print("running policy iteration")
steps = agent.iterate_policy(env, tol)

print(agent.get_value_function_pretty())

print("we can see that running policy iteration once doesn't give us the optimal policy/result")

# numpy coordinates are the transpose of graphics coordinates
""" random_vf = np.array([[3.3, 8.8, 4.4, 5.3, 1.5],
                      [1.5, 3.0, 2.3, 1.9, 0.5],
                      [0.1, 0.7, 0.7, 0.4, -0.4],
                      [-1.0, -0.4, -0.4, -0.6, -1.2],
                      [-1.9, -1.3, -1.2, -1.4, -2.0]
                     ]).T
agent.set_value_function(random_vf)
agent.set_policy((4, 0), {GridWorldActions.left: 1.0})

print("GridWorld value functions") """

""" print("\n Random policy")
print(agent.get_value_function_pretty())
steps = agent.evaluate_policy(env, 0.01)

print("\n Switching to random policy with greedy top-right state")
print(agent.get_value_function_pretty())
print(steps)

# use optimal policy
for i in range(5):
    for j in range(5):
        agent.set_policy((i, j), OPTIMAL_POLICY[i, j])

print("\nSwitching to optimal policy")
steps = agent.evaluate_policy(env, 0.01)
print(agent.get_value_function_pretty())
print(steps) """

env.close
