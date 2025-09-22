import gymnasium as gym
from gymnasium_env.envs import GridWorldNoTargetEnv

env = gym.make("gymnasium_env/GridWorldNoTarget-v0", render_mode="human")

obs, info = env.reset()
for t in range(30):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {t}, action={action}, obs={obs}, reward={reward}")
env.close()