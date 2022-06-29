# Other core modules
import copy
import functools

from gym_cooking.cooking_world.cooking_world import CookingWorld
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_book.recipe_drawer import RECIPES, NUM_GOALS

import numpy as np
from collections import namedtuple, defaultdict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

import gym
from gym.spaces import Discrete, Box, MultiBinary, Dict


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")
COLORS = ['blue', 'magenta', 'yellow', 'green']


def env(level, num_agents, record, max_steps, recipes, obs_spaces):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = CookingEnvironment(level, num_agents, record, max_steps, recipes, obs_spaces)
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


parallel_env = parallel_wrapper_fn(env)


class CookingEnvironment(AECEnv):
    """Environment object for Overcooked."""

    metadata = {
        "render_modes": ["human"],
        "name": "cooking_zoo",
        "is_parallelizable": True}

    def __init__(self, level, num_agents, record, max_steps, recipes, obs_spaces=["numeric"], allowed_objects=None):
        super().__init__()

        self.allowed_obs_spaces = ["symbolic", "numeric", "simple"]
        assert len(set(obs_spaces + self.allowed_obs_spaces)) == 3, \
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
        self.world = CookingWorld()
        self.seed()
        self.recipes = recipes
        self.game = None
        self.recipe_graphs = [RECIPES[recipe]() for recipe in recipes]

        self.termination_info = ""
        self.world.load_level(level=self.level, num_agents=num_agents)
        self.graph_representation_length = sum([tup[1] for tup in GAME_CLASSES_STATE_LENGTH]) + self.num_agents
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
        self.current_tensor_observation = dict(zip(self.agents, [np.zeros((self.world.width, self.world.height,
                                                                           self.graph_representation_length))
                                                                 for _ in self.agents]))

    def seed(self, seed=None):
        return self.world.seed(seed)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        numeric_obs_space = {
            'symbolic_observation': Box(low=0,
                                        high=10,
                                        shape=(self.world.width, 
                                               self.world.height,
                                               self.graph_representation_length),
                                        dtype=np.int32
                                        ),
             'agent_location': Box(low=0,
                                   high=max(self.world.width, 
                                            self.world.height),
                                    shape=(2,)
                                    ),
             'goal_vector': MultiBinary(NUM_GOALS)
             }
        # TODO I don't really understand what the old code (above) was doing (in what sense is this
        # the observation space of functions returned by the 'observe' function?)
        # Simple obs space
        simple_high = np.array([
            self.world.width, self.world.height, # cut_board_loc,
            self.world.width, self.world.height, # deliver_square_loc,
            self.world.width, self.world.height, # plate_loc,
            self.world.width, self.world.height, # tomato_loc,
            self.world.width, self.world.height, # chopped_tomato_loc,
            *[self.world.width, self.world.height, 1, 1, 1, 1] * len(self.possible_agents)
            ])
        simple_obs_space = Box(
            low=-1,
            high=simple_high,
            shape=(10+6*len(self.possible_agents),)
            )
        if self.obs_spaces[0] == "simple":
            return simple_obs_space
        else:
            return Dict(numeric_obs_space)

    @property
    def observation_spaces(self):
        return {agent: self.observation_space(agent)
                for agent in self.possible_agents}


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Discrete(6)

    @property
    def action_spaces(self):
        return {agent: self.action_space(agent)
                for agent in self.possible_agents}

    def set_filename(self):
        self.filename = f"{self.level}_agents{self.num_agents}"

    def state(self):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        self.world = CookingWorld()
        self.seed(seed)
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

        self.current_tensor_observation = {agent: self.get_tensor_representation(agent)
                                           for agent in self.agents}
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
        for agent in self.agents:
            self.current_tensor_observation[agent] = self.get_tensor_representation(agent)

        info = {"t": self.t, "termination_info": self.termination_info}

        done, rewards, goals = self.compute_rewards()
        for idx, agent in enumerate(self.agents):
            self.dones[agent] = done
            self.rewards[agent] = rewards[idx]
            self.infos[agent] = info

    def observe(self, agent):
        observation = []
        if "numeric" in self.obs_spaces:
            num_observation = {'numeric_observation': self.current_tensor_observation[agent],
                               'agent_location': np.asarray(self.world_agent_mapping[agent].location, np.int32),
                               'goal_vector': self.recipe_mapping[agent].goals_completed(NUM_GOALS)}
            observation.append(num_observation)
        if "symbolic" in self.obs_spaces:
            objects = defaultdict(list)
            objects.update(self.world.world_objects)
            objects["Agent"] = self.world.agents
            sym_observation = copy.deepcopy(objects)
            observation.append(sym_observation)
        if "simple" in self.obs_spaces:
            simple_observation = self.get_simple_representation(agent)
            observation.append(simple_observation)
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
            rewards[idx] = (sum(goals_before) - sum(open_goals[idx]) + bonus) * 10
            if rewards[idx] < 0:
                print(f"Goals before: {goals_before}")
                print(f"Goals after: {open_goals}")

        if all((recipe.completed() for recipe in self.recipe_graphs)):
            self.termination_info = "Terminating because all deliveries were completed"
            done = True
        return done, rewards, open_goals

    def get_simple_representation(self, agent):
        # NB: only works when there's one of each object!
        # cutting board location
        cut_board_loc = self.world.world_objects["CutBoard"][0].location
        # deliver square location
        deliver_square_loc = self.world.world_objects["DeliverSquare"][0].location
        # plate location
        plate_loc = self.world.world_objects["Plate"][0].location
        # tomato location
        tomato = self.world.world_objects["Tomato"][0]
        if tomato.chop_state == ChopFoodStates.CHOPPED:
            tomato_loc = (-1, -1)
            chopped_tomato_loc =  tomato.location
        else:
            tomato_loc =  tomato.location
            chopped_tomato_loc = (-1, -1)
        # agent locations
        ego_agent = self.world_agent_mapping[agent]
        other_agents = [agent_obj
                        for agent_id, agent_obj in self.world_agent_mapping.items()
                        if agent_id != agent]
        ego_agent_location = ego_agent.location
        ego_agent_orientation = np.zeros(4)
        ego_agent_orientation[ego_agent.orientation-1] = 1
        other_agent_info = []
        for other_agent in other_agents:
            other_agent_location = other_agent.location
            other_agent_orientation = np.zeros(4)
            other_agent_orientation[other_agent.orientation-1] = 1
            other_agent_info.extend((other_agent_location, other_agent_orientation))
        return np.concatenate([
            cut_board_loc,
            deliver_square_loc,
            plate_loc,
            tomato_loc,
            chopped_tomato_loc,
            ego_agent_location,
            ego_agent_orientation,
            *other_agent_info
            ])


    def get_tensor_representation(self, agent):
        tensor = np.zeros(
            (self.world.width, self.world.height, self.graph_representation_length + len(self.world.agents)))
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        idx = 0
        for game_class in GAME_CLASSES:
            if game_class is Agent:
                continue
            stateful_class = self.get_stateful_class(game_class)
            if stateful_class:
                n = 1
                for obj in objects[ClassToString[game_class]]:
                    # TODO carrot temp hack
                    if game_class is Carrot:
                        representation = [
                            int(obj.chop_state != ChopFoodStates.CHOPPED),
                            int(obj.chop_state == ChopFoodStates.CHOPPED),
                            obj.current_progress
                            ]
                    else:
                        representation = self.handle_stateful_class_representation(obj, stateful_class)
                    n = len(representation)
                    x, y = obj.location
                    for i in range(n):
                        tensor[x, y, idx + i] += representation[i]
                idx += n
            else:
                for obj in objects[ClassToString[game_class]]:
                    x, y = obj.location
                    tensor[x, y, idx] += 1
                idx += 1
        ego_agent = self.world_agent_mapping[agent]
        x, y = ego_agent.location
        # location map for all agents, location maps for separate agent and four orientation maps shared
        # between all agents
        tensor[x, y, idx] = 1
        tensor[x, y, idx + 1] = 1
        tensor[x, y, idx + self.num_agents + ego_agent.orientation] = 1

        agent_idx = 1
        for world_agent in self.world.agents:
            if agent != world_agent:
                x, y = world_agent.location
                # location map for all agents, location maps for separate agent and four orientation maps shared
                # between all agents
                tensor[x, y, idx] = 1
                tensor[x, y, idx + agent_idx + 1] = 1
                tensor[x, y, idx + self.num_agents + world_agent.orientation] = 1
                agent_idx += 1
        return tensor

    def get_agent_names(self):
        return [agent.name for agent in self.world.agents]

    def render(self, mode='human'):
        pass

    @staticmethod
    def get_stateful_class(game_class):
        for stateful_class in STATEFUL_GAME_CLASSES:
            if issubclass(game_class, stateful_class):
                return stateful_class
        return None

    @staticmethod
    def handle_stateful_class_representation(obj, stateful_class):
        # TODO carrot is both, so really this logic isn't great.
        if stateful_class is ChopFood:
            return [
                int(obj.chop_state != ChopFoodStates.CHOPPED),
                int(obj.chop_state == ChopFoodStates.CHOPPED)
                ]
        if stateful_class is BlenderFood:
            return [obj.current_progress]
        raise ValueError(f"Could not process stateful class {stateful_class}")
