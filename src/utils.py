import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from .env import GridWorldEnv

def epsilon_greedy(q_values, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)  # four actions (0-3)
    else:
        return torch.argmax(q_values).item()

def create_q_table(dqn):
    """
    Create a Q-table from a trained DQN model.
    Returns:
        q_table: numpy array of shape (num_states, num_actions)
                 where num_states = n_rows * n_cols
    """
    env = GridWorldEnv()
    n_rows, n_cols = env.n_rows, env.n_cols
    action_dim = 4  # Assuming 4 actions;

    q_table = np.zeros((n_rows * n_cols, action_dim))

    for r in range(n_rows):
        for c in range(n_cols):
            pos = (r, c)
            state_vec = env.encode(pos)
            with torch.no_grad():
                state_tensor = torch.tensor(state_vec).unsqueeze(0).float()  # shape: 1 x state_dim
                q_values = dqn(state_tensor).cpu().numpy().squeeze()  # shape: (action_dim,)
            state_idx = r * n_cols + c
            q_table[state_idx, :] = q_values

    return q_table

def plot_state_values(dqn):
    """
    Plot max Q-values per state as a heatmap, with numeric values on each cell.
    This queries the DQN directly per state.
    """
    env = GridWorldEnv()
    grid = np.zeros((env.n_rows, env.n_cols))

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            pos = (r, c)
            state = env.encode(pos)
            with torch.no_grad():
                value = dqn(torch.tensor(state).unsqueeze(0).float()).max(1)[0].item()
            grid[r, c] = value

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='coolwarm', origin='upper')

    # Add numeric Q-values on top of heatmap
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            plt.text(c, r, f"{grid[r, c]:.1f}", ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(label='State value (max Q)')
    plt.title("State Value Heatmap (DQN Agent)")
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.xticks(range(env.n_cols))
    plt.yticks(range(env.n_rows))
    plt.show()

def plot_q_table(q_table):
    """
    Plot a heatmap of the max Q-values per state from a given Q-table.
    Assumes q_table shape is (num_states, num_actions).
    """
    env = GridWorldEnv()
    n_rows, n_cols = env.n_rows, env.n_cols

    max_q_values = np.max(q_table, axis=1).reshape((n_rows, n_cols))

    plt.figure(figsize=(6, 6))
    plt.imshow(max_q_values, cmap='coolwarm', origin='upper')

    for r in range(n_rows):
        for c in range(n_cols):
            plt.text(c, r, f"{max_q_values[r, c]:.1f}", ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(label='Max Q-value per state')
    plt.title("State Value Heatmap from Q-Table")
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.xticks(range(n_cols))
    plt.yticks(range(n_rows))
    plt.show()
