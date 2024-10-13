from gymnasium.envs.registration import register

# Simple 5x5
register(
    id="simple-5x5",
    entry_point="gym_env.envs:MazeEnv5x5",
    max_episode_steps=2000,
)

# Simple 5x5 with two goal states
register(
    id="simple-5x5-2",
    entry_point="gym_env.envs:MazeEnv5x5_2G",
    max_episode_steps=2000,
)

# Simple 7x7 with two goal states
register(
    id="simple-7x7-2",
    entry_point="gym_env.envs:MazeEnv7x7_2G",
    max_episode_steps=2000,
)

# Hairpin
register(
    id="hairpin-14x14",
    entry_point="gym_env.envs:MazeEnvHairpin",
    max_episode_steps=2000,
)

# Tolman detour task
register(
    id="tolman-9x9-nb",
    entry_point="gym_env.envs:MazeEnvTolmanNB",
    max_episode_steps=2000,
)

register(
    id="tolman-9x9-b",
    entry_point="gym_env.envs:MazeEnvTolmanB",
    max_episode_steps=2000,
)

# Tolman latent task
register(
    id="tolman-latent",
    entry_point="gym_env.envs:MazeEnvTolmanLatentOG",
    max_episode_steps=2000,
)

# Tolman latent task from Russek et al.
register(
    id="tolman-10x10-latent",
    entry_point="gym_env.envs:MazeEnvTolmanLatent",
    max_episode_steps=2000,
)

# Tolman latent task new goal
register(
    id="tolman-10x10-latent-new-goal",
    entry_point="gym_env.envs:MazeEnvTolmanLatentNewGoal",
    max_episode_steps=2000,
)

# 10x10 two goals
register(
    id="maze-10x10-two-goal",
    entry_point="gym_env.envs:MazeEnv10x10_2G",
    max_episode_steps=2000,
)

# 10x10 with different goal states
register(
    id="maze-10x10-G0",
    entry_point="gym_env.envs:MazeEnv10x10_G0",
    max_episode_steps=2000,
)

register(
    id="maze-10x10-G1",
    entry_point="gym_env.envs:MazeEnv10x10_G1",
    max_episode_steps=2000,
)

register(
    id="maze-10x10-G2",
    entry_point="gym_env.envs:MazeEnv10x10_G2",
    max_episode_steps=2000,
)

register(
    id="maze-10x10-G3",
    entry_point="gym_env.envs:MazeEnv10x10_G3",
    max_episode_steps=2000,
)

# Four room task
register(
    id="four_room_tr",
    entry_point="gym_env.envs:MazeEnv4RoomTR",
    max_episode_steps=2000,
)

register(
    id="four_room_br",
    entry_point="gym_env.envs:MazeEnv4RoomBR",
    max_episode_steps=2000,
)

# 15x15
register(
    id="simple-15x15",
    entry_point="gym_env.envs:MazeEnv15x15",
    max_episode_steps=2000,
)

# 15x15 new goal
register(
    id="simple-15x15-new-goal",
    entry_point="gym_env.envs:MazeEnv15x15NewGoal",
    max_episode_steps=2000,
)

# 15x15 with different goal states
register(
    id="maze-15x15-G0",
    entry_point="gym_env.envs:MazeEnv15x15_G0",
    max_episode_steps=2000,
)

register(
    id="maze-15x15-G1",
    entry_point="gym_env.envs:MazeEnv15x15_G1",
    max_episode_steps=2000,
)

# Open Field
register(
    id="open-field-no-goal",
    entry_point="gym_env.envs:MazeEnvOpenFieldNoGoal",
    max_episode_steps=2000,
)

register(
    id="open-field-center-goal",
    entry_point="gym_env.envs:MazeEnvOpenFieldCenterGoal",
    max_episode_steps=2000,
)

# Carpenter et al. (2015) maze
register(
    id="carpenter",
    entry_point="gym_env.envs:MazeEnvCarpenter",
    max_episode_steps=2000,
)