import time

from cooking_zoo.environment.cooking_env import parallel_env
from cooking_zoo.environment.manual_policy import ManualPolicy
from cooking_zoo.cooking_agents.cooking_agent import CookingAgent


num_agents = 2
max_steps = 400
render_mode = "human"
obs_spaces = ["feature_vector", "symbolic"]
action_scheme = "scheme3"
meta_file = "example"
level = "coop_test"
recipes = ["TomatoLettuceSalad", "CarrotBanana"]
end_condition_all_dishes = True
agent_visualization = ["robot", "human"]
reward_scheme = {"recipe_reward": 20, "max_time_penalty": -5, "recipe_penalty": -40, "recipe_node_reward": 0}
agent_respawn_rate = 0.5
grace_period = 1
agent_despawn_rate = 0.5

cooking_agent = CookingAgent(recipes[1], "agent-2")

env = parallel_env(level=level, meta_file=meta_file, num_agents=num_agents, max_steps=max_steps, recipes=recipes,
                   agent_visualization=agent_visualization, obs_spaces=obs_spaces,
                   end_condition_all_dishes=end_condition_all_dishes, action_scheme=action_scheme, render_mode=render_mode,
                   reward_scheme=reward_scheme, agent_despawn_rate=agent_despawn_rate,
                   agent_respawn_rate=agent_respawn_rate, grace_period=grace_period)

observations, info = env.reset()

env.render()

action_space = env.action_space("player_0")

manual_policy = ManualPolicy(env, agent_id="player_0")

terminations = {"player_0": False, "player_1": False}
truncations = {"player_0": False, "player_1": False}

while not all(terminations.values()):
    actions = {}
    if "player_0" in observations and not terminations["player_0"] and not truncations["player_0"]:
        actions["player_0"] = manual_policy("player_0")
    else:
        time.sleep(0.1)
    if "player_1" in observations and not terminations["player_1"] and not truncations["player_1"]:
        actions["player_1"] = cooking_agent.step(observations["player_1"])
    # action = {"player_0": manual_policy("player_0"), "player_1": cooking_agent.step(observations["player_1"])}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(infos)
    env.render()
