from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

register(
    id="gymnasium_env/OurGridWorld-v0",
    entry_point="gymnasium_env.envs:OurGridWorldEnv",
)
