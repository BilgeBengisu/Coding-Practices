# Run `pip install "gymnasium[classic-control]"` for this example.
import numpy as np
import gymnasium as gym
import gymnasium_env
from optimal_policy import OPTIMAL_POLICY
from gymnasium_env.agents.grid_world_agent import GridWorldAgent
from gymnasium_env.agents.grid_world_actions import GridWorldActions

agent = GridWorldAgent(4, 1.0)
env = gym.make("gymnasium_env/SmallGridWorld-v0", render_mode=None, agent=agent)
observation, info = env.reset()

# numpy coordinates are the transpose of graphics coordinates
zero_vf = np.array([[0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]
                   ]).T
agent.set_value_function(zero_vf)

# Duplicate the value functions of Figure 4.1:
# The value functions of the random policy after each step of policy evalution.

# The wrapped environment doesn't have access to newly defined methods.
uenv = env.unwrapped
print("SmallGridWorld random policy value functions after k steps of policy evaluation")
eps = agent.evaluate_policy_once(uenv)
print(f"k=1, epsilon={eps:.2f}:")
print(agent.get_value_function_pretty())
stable = agent.improve_policy_once(uenv)
print(f"Greedy policy for new value function (Stable = {stable}):")
print(agent.get_policy_pretty())

print(f"\nResetting to random policy. Keeping new value function.")
agent.set_policy_random()
eps = agent.evaluate_policy_once(uenv)
print(f"k=2, epsilon={eps:.2f}:")
print(agent.get_value_function_pretty())
stable = agent.improve_policy_once(uenv)
print(f"Greedy policy for new value function (Stable = {stable}):")
print(agent.get_policy_pretty())

print(f"\nResetting to random policy. Keeping new value function.")
agent.set_policy_random()
eps = agent.evaluate_policy_once(uenv)
print(f"k=3, epsilon={eps:.2f}:")
print(agent.get_value_function_pretty())
stable = agent.improve_policy_once(uenv)
print(f"Greedy policy for new value function (Stable = {stable}):")
print(agent.get_policy_pretty())

print(f"\nResetting to random policy. Keeping new value function.")
agent.set_policy_random()
for i in range(4,10):
    eps = agent.evaluate_policy_once(uenv)
print(f"k=10, epsilon={eps:.2f}:")
print(agent.get_value_function_pretty())
stable = agent.improve_policy_once(uenv)
print(f"Greedy policy for new value function (Stable = {stable}):")
print(agent.get_policy_pretty())

print(f"\nResetting to random policy. Keeping new value function.")
agent.set_policy_random()
steps = agent.evaluate_policy(uenv, 0.01)
print(f"k={10+steps}, epsilon=0.01:")
print(agent.get_value_function_pretty())
stable = agent.improve_policy_once(uenv)
print(f"Greedy policy for new value function (Stable = {stable}):")
print(agent.get_policy_pretty())

print(f"\nFinally, policy iteration.")
steps = agent.iterate_policy(uenv, 0.01) 
print(f"k={steps}:")
print(agent.get_value_function_pretty())
print(agent.get_policy_pretty())

env.close
