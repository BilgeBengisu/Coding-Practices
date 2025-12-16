from gymnasium.envs.registration import register #https://gymnasium.farama.org/api/registry/


register(id='WindyGridWorld-v0',
         entry_point='envs.windy_gridworld_env:WindyGridWorldEnv',
        )

register(id='WindyGridWorld-v1',
         entry_point='envs.king_windy_gridworld_env:KingWindyGridWorldEnv',
        )