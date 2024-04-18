import gymnasium as gym

gym.envs.registration.register(
    id='Vizdoom-v0',
    entry_point='src.game.gymnasium_wrapper:VizDOOM',
    max_episode_steps=1000,
)

