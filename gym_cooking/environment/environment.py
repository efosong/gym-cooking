from gym_cooking.environment import cooking_zoo
from gym.utils import seeding

import gym


class GymCookingEnvironment(gym.Env):
    """Environment object for Overcooked."""

    metadata = {'render.modes': ['human'], 'name': "cooking_zoo"}

    def __init__(self, level, record, max_steps, recipe, obs_spaces=None, action_scheme="scheme1", ghost_agents=0):
        super().__init__()
        self.num_agents = 1
        self.zoo_env = cooking_zoo.parallel_env(level=level, num_agents=self.num_agents, record=record,
                                                max_steps=max_steps, recipes=[recipe], obs_spaces=obs_spaces,
                                                action_scheme=action_scheme, ghost_agents=ghost_agents)
        self.observation_space = self.zoo_env.observation_spaces["player_0"]
        self.action_space = self.zoo_env.action_spaces["player_0"]

    def step(self, action):
        converted_action = {"player_0": action}
        obs, reward, done, info = self.zoo_env.step(converted_action)
        return obs["player_0"], reward["player_0"], done["player_0"], info["player_0"]

    def reset(self):
        return self.zoo_env.reset()["player_0"]

    def render(self, mode='human'):
        pass


class EPyMARLGymCooking(gym.Env):

    def __init__(
        self,
        level,
        num_agents,
        record,
        max_steps,
        recipes,
        obs_spaces,
        action_scheme="full_action_scheme",
        ghost_agents=0,
        completion_reward_frac=0.2,
        time_penalty=0.0,
        ego_first=True,
    ):
        super().__init__()
        self.zoo_env = cooking_zoo.parallel_env(
            level=level,
            num_agents=num_agents,
            record=record,
            max_steps=max_steps,
            recipes=recipes,
            obs_spaces=obs_spaces,
            action_scheme=action_scheme,
            ghost_agents=ghost_agents,
            completion_reward_frac=completion_reward_frac,
            time_penalty=time_penalty,
            ego_first=ego_first,
            )
        self.observation_space = [self.zoo_env.observation_space(agent_id)
                                  for agent_id in self.zoo_env.possible_agents]
        self.action_space = [self.zoo_env.action_space(agent_id)
                             for agent_id in self.zoo_env.possible_agents]
        self.n_agents = len(self.zoo_env.possible_agents)

    def step(self, action):
        # we expect a list of actions
        self.zoo_env.possible_agents
        pz_action = dict(zip(self.zoo_env.possible_agents, action))
        pz_obs, pz_rew, pz_term, pz_trunc, pz_info = self.zoo_env.step(pz_action)
        obs = list(pz_obs[agent_id] for agent_id in self.zoo_env.possible_agents)
        rew = list(pz_rew[agent_id] for agent_id in self.zoo_env.possible_agents)
        done = list(pz_term[agent_id] or pz_trunc[agent_id]
                     for agent_id in self.zoo_env.possible_agents)
        # TODO not passing the infos for now
        info = {}
        return obs, rew, done, info

    def reset(self):
        pz_obs, pz_info = self.zoo_env.reset()
        obs = list(pz_obs[agent_id] for agent_id in self.zoo_env.possible_agents)
        return obs
