# Other core modules
import copy

from gym_cooking.cooking_world.cooking_world import CookingWorld
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_world.actions import *
from gym_cooking.cooking_book.recipe_drawer import RECIPES, NUM_GOALS

import numpy as np
from collections import namedtuple, defaultdict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

import gym

CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")
COLORS = ['blue', 'magenta', 'yellow', 'green']


def env(level, num_agents, record, max_steps, recipes, obs_spaces=None, action_scheme="scheme1", ghost_agents=0):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = CookingEnvironment(level, num_agents, record, max_steps, recipes, obs_spaces,
                                  action_scheme=action_scheme, ghost_agents=ghost_agents)
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


parallel_env = parallel_wrapper_fn(env)


class CookingEnvironment(AECEnv):
    """Environment object for Overcooked."""

    metadata = {'render.modes': ['human'], 'name': "cooking_zoo"}

    action_scheme_map = {"scheme1": ActionScheme1, "scheme2": ActionScheme2, "scheme3": ActionScheme3}

    def __init__(self, level, num_agents, record, max_steps, recipes, obs_spaces=None, allowed_objects=None,
                 action_scheme="scheme1", ghost_agents=0):
        super().__init__()

        obs_spaces = obs_spaces or ["numeric"]
        self.allowed_obs_spaces = ["symbolic", "numeric", "numeric_main", "feature_vector"]
        self.action_scheme = action_scheme
        self.action_scheme_class = self.action_scheme_map[self.action_scheme]
        assert len(set(obs_spaces + self.allowed_obs_spaces)) == 4, \
            f"Selected invalid obs spaces. Allowed {self.allowed_obs_spaces}"
        assert len(obs_spaces) != 0, f"Please select an observation space from: {self.allowed_obs_spaces}"
        self.obs_spaces = obs_spaces
        self.allowed_objects = allowed_objects or []
        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]

        self.level = level
        self.record = record
        self.max_steps = max_steps
        self.t = 0
        self.filename = ""
        self.set_filename()
        self.world = CookingWorld(self.action_scheme_class)
        self.recipes = recipes
        self.game = None
        self.recipe_graphs = [RECIPES[recipe]() for recipe in recipes]
        self.ghost_agents = ghost_agents

        self.termination_info = ""
        self.world.load_level(level=self.level, num_agents=num_agents)
        self.graph_representation_length = sum([cls.state_length() for cls in GAME_CLASSES])
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        objects["Agent"] = self.world.agents
        feat_vec_l = sum([obj.feature_vector_length() for cls in GAME_CLASSES for obj in objects[ClassToString[cls]]])
        agent_feature_length = StringToClass["Agent"].feature_vector_length()
        self.feature_vector_representation_length = feat_vec_l + (agent_feature_length * self.ghost_agents)

        numeric_obs_space = {'symbolic_observation': gym.spaces.Box(low=0, high=10,
                                                                    shape=(self.world.width, self.world.height,
                                                                           self.graph_representation_length),
                                                                    dtype=np.int32),
                             'agent_location': gym.spaces.Box(low=0, high=max(self.world.width, self.world.height),
                                                              shape=(2,)),
                             'goal_vector': gym.spaces.MultiBinary(NUM_GOALS)}
        self.feature_obs_space = gym.spaces.Box(low=-1, high=1,
                                                shape=(self.feature_vector_representation_length,))
        self.numeric_main_obs_space = gym.spaces.Box(low=0, high=10, shape=(self.world.width, self.world.height,
                                                                            self.graph_representation_length))
        obs_space_dict = {"numeric": numeric_obs_space, "numeric_main": self.numeric_main_obs_space,
                          "feature_vector": self.feature_obs_space}
        self.observation_spaces = {agent: obs_space_dict[obs_spaces[0]] for agent in self.possible_agents}
        self.action_spaces = {agent: gym.spaces.Discrete(len(self.action_scheme_class.ACTIONS))
                              for agent in self.possible_agents}
        self.has_reset = True

        self.recipe_mapping = dict(zip(self.possible_agents, self.recipe_graphs))
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_tensor_observation = np.zeros((self.world.width, self.world.height,
                                                    self.graph_representation_length))

    def set_filename(self):
        self.filename = f"{self.level}_agents{self.num_agents}"

    def state(self):
        pass

    def reset(self):
        # self.world = CookingWorld(self.action_scheme_class)
        self.t = 0

        # For tracking data during an episode.
        self.termination_info = ""

        # Load world & distances.
        self.world.load_level(level=self.level, num_agents=self.num_agents)

        for recipe in self.recipe_graphs:
            recipe.update_recipe_state(self.world)

        # if self.record:
        #     self.game = GameImage(
        #         filename=self.filename,
        #         world=self.world,
        #         record=self.record)
        #     self.game.on_init()
        #     self.game.save_image_obs(self.t)
        # else:
        #     self.game = None

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Get an image observation
        # image_obs = self.game.get_image_obs()
        self.recipe_mapping = dict(zip(self.possible_agents, self.recipe_graphs))
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))

        self.current_tensor_observation = self.get_tensor_representation()
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []

    def close(self):
        return

    def step(self, action):
        agent = self.agent_selection
        self.accumulated_actions.append(action)
        for idx, agent in enumerate(self.agents):
            self.rewards[agent] = 0
        if self._agent_selector.is_last():
            self.accumulated_step(self.accumulated_actions)
            self.accumulated_actions = []
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        # translated_actions = [action_translation_dict[actions[f"player_{idx}"]] for idx in range(len(actions))]
        self.world.perform_agent_actions(self.world.agents, actions)

        # Visualize.
        if self.record:
            self.game.on_render()

        if self.record:
            self.game.save_image_obs(self.t)

        # Get an image observation
        # image_obs = self.game.get_image_obs()
        self.current_tensor_observation = self.get_tensor_representation()

        info = {"t": self.t, "termination_info": self.termination_info}

        done, rewards, goals = self.compute_rewards()
        for idx, agent in enumerate(self.agents):
            self.dones[agent] = done
            self.rewards[agent] = rewards[idx]
            self.infos[agent] = info

    def observe(self, agent):
        observation = []
        if "numeric" in self.obs_spaces:
            num_observation = {'symbolic_observation': self.current_tensor_observation,
                               'agent_location': np.asarray(self.world_agent_mapping[agent].location, np.int32),
                               'goal_vector': self.recipe_mapping[agent].goals_completed(NUM_GOALS)}
            observation.append(num_observation)
        if "symbolic" in self.obs_spaces:
            objects = defaultdict(list)
            objects.update(self.world.world_objects)
            objects["Agent"] = self.world.agents
            sym_observation = copy.deepcopy(objects)
            observation.append(sym_observation)
        if "numeric_main" in self.obs_spaces:
            observation.append(self.current_tensor_observation)
        if "feature_vector" in self.obs_spaces:
            observation.append(self.get_feature_vector(agent))
        returned_observation = observation if not len(observation) == 1 else observation[0]
        return returned_observation

    def compute_rewards(self):
        done = False
        rewards = [0] * len(self.recipes)
        open_goals = [[0]] * len(self.recipes)
        # Done if the episode maxes out
        if self.t >= self.max_steps and self.max_steps:
            self.termination_info = f"Terminating because passed {self.max_steps} timesteps"
            done = True

        for idx, recipe in enumerate(self.recipe_graphs):
            goals_before = recipe.goals_completed(NUM_GOALS)
            recipe.update_recipe_state(self.world)
            open_goals[idx] = recipe.goals_completed(NUM_GOALS)
            bonus = recipe.completed() * 0.1
            rewards[idx] = (sum(goals_before) - sum(open_goals[idx]) + bonus) * 5

            rewards[idx] -= (5 / self.max_steps)

            objects_to_seek = recipe.get_objects_to_seek()
            # if objects_to_seek:
            #     distances = []
            #     for cls in objects_to_seek:
            #         world_objects = self.world.world_objects[ClassToString[cls]]
            #         min_distance = min([abs(self.world.agents[idx].location[0] - obj.location[0]) / self.world.height +
            #                             abs(self.world.agents[idx].location[1] - obj.location[1]) / self.world.width
            #                             for obj in world_objects])
            #         distances.append(min_distance)
            #
            #     rewards[idx] -= min(distances)

        for idx, agent in enumerate(self.world.agents):
            if not agent.interacts_with:
                rewards[idx] -= 0.01

        if all((recipe.completed() for recipe in self.recipe_graphs)):
            self.termination_info = "Terminating because all deliveries were completed"
            done = True
        return done, rewards, open_goals

    def get_feature_vector(self, agent):
        feature_vector = []
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        objects["Agent"] = self.world.agents
        x, y = self.world_agent_mapping[agent].location
        for cls in GAME_CLASSES:
            for obj in objects[ClassToString[cls]]:
                features = list(obj.feature_vector_representation())
                if features and obj is not self.world_agent_mapping[agent]:
                    features[0] = (features[0] - x) / self.world.width
                    features[1] = (features[1] - y) / self.world.height
                if obj is self.world_agent_mapping[agent]:
                    features[0] = features[0] / self.world.width
                    features[1] = features[1] / self.world.height
                feature_vector.extend(features)
        for idx in range(self.ghost_agents):
            features = self.world_agent_mapping[agent]
            features[0] = 0
            features[1] = 0

        return np.array(feature_vector)

    def get_tensor_representation(self):
        tensor = np.zeros((self.world.width, self.world.height, self.graph_representation_length))
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        objects["Agent"] = self.world.agents
        state_idx = 0
        for cls in GAME_CLASSES:
            for obj in objects[ClassToString[cls]]:
                x, y = obj.location
                for idx, value in enumerate(obj.numeric_state_representation()):
                    tensor[x, y, state_idx + idx] = value
            state_idx += cls.state_length()
        return tensor

    def get_agent_names(self):
        return [agent.name for agent in self.world.agents]

    def render(self, mode='human'):
        pass
