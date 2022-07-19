from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_book.recipe import Recipe, RecipeNode
from copy import deepcopy


def id_num_generator():
    num = 0
    while True:
        yield num
        num += 1


id_generator = id_num_generator()

#  Basic food Items
# root_type, id_num, parent=None, conditions=None, contains=None
ChoppedLettuce = RecipeNode(root_type=Lettuce, id_num=next(id_generator), name="Lettuce",
                            conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Lettuce, Cutboard])
ChoppedOnion = RecipeNode(root_type=Onion, id_num=next(id_generator), name="Onion",
                          conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Onion, Cutboard])
ChoppedTomato = RecipeNode(root_type=Tomato, id_num=next(id_generator), name="Tomato",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)], objects_to_seek=[Tomato, Cutboard])
# MashedCarrot = RecipeNode(root_type=Carrot, id_num=next(id_generator), name="Carrot",
#                           conditions=[("blend_state", BlenderFoodStates.MASHED)])

# Salad Plates
TomatoSaladPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedTomato], objects_to_seek=[Tomato, Plate])
TomatoLettucePlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedTomato, ChoppedLettuce], objects_to_seek=[Tomato, Plate, Lettuce, Plate])
TomatoLettuceOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                     contains=[ChoppedTomato, ChoppedLettuce, ChoppedOnion],
                                     objects_to_seek=[Tomato, Plate, Lettuce, Plate, Onion, Plate])
# CarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
#                          contains=[MashedCarrot])

# Delivered Salads
TomatoSalad = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare", conditions=None,
                         contains=[TomatoSaladPlate], objects_to_seek=[Plate, Deliversquare])
TomatoLettuceSalad = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare",
                                conditions=None, contains=[TomatoLettucePlate]
                                , objects_to_seek=[Plate, Deliversquare])
TomatoLettuceOnionSalad = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare",
                                     conditions=None, contains=[TomatoLettuceOnionPlate],
                                     objects_to_seek=[Plate, Deliversquare])
# MashedCarrot = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name="Deliversquare",
#                           conditions=None, contains=[CarrotPlate])

floor = RecipeNode(root_type=Floor, id_num=next(id_generator), name="Floor", conditions=None, contains=[])
no_recipe_node = RecipeNode(root_type=Deliversquare, id_num=next(id_generator), name='Deliversquare', conditions=None, contains=[floor], objects_to_seek=[])

# this one increments one further and is thus the amount of ids we have given since
# we started counting at zero.
NUM_GOALS = next(id_generator)

RECIPES = {"TomatoSalad": lambda: deepcopy(Recipe(TomatoSalad, NUM_GOALS)),
           "TomatoLettuceSalad": lambda: deepcopy(Recipe(TomatoLettuceSalad, NUM_GOALS)),
           "TomatoLettuceOnionSalad": lambda: deepcopy(Recipe(TomatoLettuceOnionSalad, NUM_GOALS)),
           # "MashedCarrot": lambda: deepcopy(Recipe(MashedCarrot)),
           "no_recipe": lambda: deepcopy(Recipe(no_recipe_node, NUM_GOALS))
           }
