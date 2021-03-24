import experiment_buddy

FIRST_BRANCH_HEIGHT = .24
BRANCH_THICCNESS = 0.015
BRANCH_LENGTH = 1 / 9
MAX_BRANCHING = 10
LIGHT_WIDTH = .25
LIGHT_DIF = 250

experiment_buddy.register(locals())

tensorboard = experiment_buddy.deploy("mila", sweep_yaml="", entity='growspace')
