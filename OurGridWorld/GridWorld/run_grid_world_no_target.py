import gymnasium as gym
from gymnasium_env.envs import GridWorldNoTargetEnv

env = gym.make("gymnasium_env/GridWorldNoTarget-v0", render_mode="human")

obs, info = env.reset()
for t in range(30):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}, action={action}, obs={obs}, reward={reward}")
env.close()
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