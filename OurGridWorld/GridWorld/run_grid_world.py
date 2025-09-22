# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import gymnasium_env

# Create our training environment - a GridWorld
env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - its location
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: 'agent': array([1, 0])
# [x, y] in graphics coordinates, origin = top left

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = right, 1 = up, 2 = left, 3 = down
    # Random action for now - real agents will be smarter!
    action = env.action_space.sample()  

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: see picture
    # terminated: no termination
    # truncated: True if we hit the time limit 

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
