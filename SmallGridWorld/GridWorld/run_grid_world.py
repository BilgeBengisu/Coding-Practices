# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import gymnasium_env
from gymnasium_env.agents.grid_world_agent import GridWorldAgent
from optimal_policy import OPTIMAL_POLICY

# Give the environment a GridWorldAgent at creation
# Using a public setter is tricky, as the gymnasium OrderEnforcing object can complain
# when functions other than reset(), step(), etc are called on the environment.
agent = GridWorldAgent(5) # default random policy

# use optimal policy
for i in range(5):
    for j in range(5):
        agent.set_policy((i, j), OPTIMAL_POLICY[i, j])

# Create our training environment - a GridWorld
env = gym.make("gymnasium_env/OurGridWorld-v0", render_mode="human", agent=agent)

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - its location
# info: extra debugging information (usually not needed for basic learning)

# Example output: 'agent': array([1, 0])
# [x, y] in graphics coordinates, origin = top left

episode_over = False

while not episode_over:
    # Choose an action: 0 = right, 1 = up, 2 = left, 3 = down
    #! ask agent for action
    action = agent.get_action()  

    # Take the action and get results
    #! give location (obs["agent"?]) to agent, so agent can update itself
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: see picture
    # terminated: no termination
    # truncated: True if we hit the time limit 

    episode_over = terminated or truncated

print(f"Episode finished! Return: {agent.get_return():.2f}")
env.close()
