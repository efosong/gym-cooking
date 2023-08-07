from gym.envs.registration import register

register(id="cookingEnv-v1",
         entry_point="gym_cooking.environment:GymCookingEnvironment")
register(id="cookingZooEnv-v0",
         entry_point="gym_cooking.environment:CookingZooEnvironment")

register(id="cookingSplit-v0",
         entry_point="gym_cooking.environment:EPyMARLGymCooking",
         max_episode_steps=100,
         kwargs={
             "level": "split_room_deterministic_hard",
             "num_agents": 2,
             "max_steps": 100,
             "record": False,
             "recipes": ["TomatoSalad", "TomatoSalad"],
             "obs_spaces": ["feature_vector_nc"],
             "action_scheme": "full_action_scheme",
             "ghost_agents": 0,
             "completion_reward_frac": 0.2,
             "time_penalty": 0,
             "ego_first": True,

             }
         )
