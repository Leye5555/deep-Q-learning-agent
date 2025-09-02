# Grid parameters
N_ROWS = 5
N_COLS = 5
START_POS = (1, 0)        # [2,1] in 0-indexed (row, col)
TERMINAL_POS = (4, 4)   # [5,5] in 0-indexed
JUMP_FROM = (1, 3)  # [2,4]
JUMP_TO = (3, 3) # [4,4]
OBSTACLES = [(2, 2), (2, 3), (2,4), (3,2)] # black cells in the grid

# DQN hyperparameters
EPISODES = 300
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.985
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE = 10