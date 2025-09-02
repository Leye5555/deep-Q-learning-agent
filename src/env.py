import numpy as np
from config import N_ROWS, N_COLS, START_POS, TERMINAL_POS, JUMP_FROM, JUMP_TO, OBSTACLES
class GridWorldEnv:
    def __init__(self):
        self.n_rows =N_ROWS
        self.n_cols = N_COLS
        self.start_pos = START_POS  # [2,1] in 0-indexed (row, col)
        self.terminal_pos = TERMINAL_POS  # [5,5] in 0-indexed
        self.jump_from = JUMP_FROM  # [2,4]
        self.jump_to = JUMP_TO  # [4,4]
        self.obstacles = OBSTACLES
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        self.done = False
        return self.encode(self.agent_pos)

    def encode(self, pos):
        # One-hot encode agent position as state vector
        idx = pos[0] * self.n_cols + pos[1]
        vec = np.zeros(self.n_rows * self.n_cols)
        vec[idx] = 1
        return vec.astype(np.float32)

    def step(self, action):
        if self.done:
            raise Exception("Episode is done, reset environment.")

        row, col = self.agent_pos
        # 0: North, 1: South, 2: East, 3: West
        delta = [(-1, 0), (1, 0), (0, 1), (0, -1)][action]
        new_row, new_col = row + delta[0], col + delta[1]
        next_pos = (new_row, new_col)

        reward = -1  # default reward

        # Hit wall
        if not (0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols):
            next_pos = self.agent_pos  # stay in place
            reward = -1

        # Obstacle
        if next_pos in self.obstacles:
            next_pos = self.agent_pos
            reward = -1

        # Special jump
        if self.agent_pos == self.jump_from:
            next_pos = self.jump_to
            reward = 5

        # Terminal
        if next_pos == self.terminal_pos:
            reward = 10
            self.done = True

        self.agent_pos = next_pos
        return self.encode(next_pos), reward, self.done, {}
