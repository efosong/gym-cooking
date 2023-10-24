from gym_cooking.environment.game.game import Game

from gym_cooking.environment import cooking_zoo

n_agents = 1
num_humans = 1
render = False

#level = 'tomato_carrot_split_med'
#level = 'tomato_carrot_split_tiny'
level = 'open_room_blender'
#level = 'split_room_deterministic_hard'
#level = 'split_room_onesided_l_deterministic'
seed = 1
record = False
max_num_timesteps = 250
#recipes = ["TomatoCarrotMash", 'TomatoCarrotMash']
#recipes = ["TomatoSalad", "TomatoSalad", "TomatoSalad", "TomatoSalad", ]
#recipes = ["TomatoSalad", "TomatoSalad"]
recipes = ["TomatoSalad"]

parallel_env = cooking_zoo.parallel_env(
    level=level,
    num_agents=n_agents,
    record=record,
    max_steps=max_num_timesteps,
    recipes=recipes,
    obs_spaces=["feature_vector_nc"],
    completion_reward_frac=0.0,
    )

action_spaces = parallel_env.action_spaces
#player_2_action_space = action_spaces["player_1"]


class CookingAgent:

    def __init__(self, action_space):
        self.action_space = action_space
        self.map = {
                "U": 4,
                "D": 3,
                "R": 2,
                "L": 1,
                "X": 0,
                "P": 5,
                "S": 6,
                "A": 7,
                }

    #def get_action(self, observation) -> int:
    #    return self.action_space.sample()
    def get_action(self, observation) -> int:
        # WALK_UP = 4
        # WALK_DOWN = 3
        # WALK_RIGHT = 2
        # WALK_LEFT = 1
        # NO_OP = 0
        # INTERACT_PRIMARY = 5
        # INTERACT_PICK_UP_SPECIAL = 6
        # EXECUTE_ACTION = 7

        action = self.map[input("action: ")]
        #return action
        return self.action_space.sample()


#cooking_agent = CookingAgent(player_2_action_space)

#game = Game(parallel_env, num_humans, [cooking_agent], max_num_timesteps)
#store = game.on_execute()
game = Game(
    env=parallel_env,
    num_humans=1, 
    #ai_policies=[cooking_agent,cooking_agent,cooking_agent],
    ai_policies=[],
    render=True, 
    record=False)
store = game.on_execute()

print("done")

# Game(env, num_humans, ai_policies, max_steps=100, render=False):
