# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import gymnasium_env
from gymnasium_env.agents.grid_world_agent import GridWorldAgent

# Give the environment a GridWorldAgent at creation
# Using a public setter is tricky, as the gymnasium OrderEnforcing object can complain
# when functions other than reset(), step(), etc are called on the environment.


## SET POLICY
# agent = GridWorldAgent(4, 1.0) # default random policy

# Create our training environment - a GridWorld
env = gym.make("gymnasium_env/SmallGridWorld-v0", render_mode="human", size=4, agent=agent)

# Reset environment to start a new episode
observation, info = env.reset()

episode_over = False

while not episode_over:
    action = agent.get_action()  

    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

print(f"Episode finished! Return: {agent.get_return():.2f}")
env.close()
