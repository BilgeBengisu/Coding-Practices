from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
    #max_episode_steps=1000 # wraps the environment in a timer class
)

register(
    id="gymnasium_env/OurGridWorld-v0",
    entry_point="gymnasium_env.envs:OurGridWorldEnv",
)

register(
    id="gymnasium_env/SmallGridWorld-v0",
    entry_point="gymnasium_env.envs:SmallGridWorldEnv",
)
