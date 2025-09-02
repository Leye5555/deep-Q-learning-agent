import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Assuming your project structure allows these imports
from src.agent import train_dqn
from src.env import GridWorldEnv
from src.utils import create_q_table

# Ensure a directory exists for the plots
if not os.path.exists('results'):
    os.makedirs('results')


def plot_rewards(rewards_log, filename='rewards_plot.png'):
    """Plots cumulative rewards per episode."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_log, label='Reward per Episode', alpha=0.7)

    # Calculate and plot rolling average
    if len(rewards_log) >= 30:
        rolling_avg = np.convolve(rewards_log, np.ones(30) / 30, mode='valid')
        plt.plot(range(29, len(rewards_log)), rolling_avg, label='30-Episode Rolling Average', color='orange')

    plt.title('Cumulative Reward per Episode During Training')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join('results', filename))
    plt.show()
    print(f"Saved reward plot to results/{filename}")


def plot_policy_map(dqn, filename='policy_map.png'):
    """Visualizes the learned policy as a grid of arrows."""
    env = GridWorldEnv()
    q_table = create_q_table(dqn)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, env.n_cols - 0.5)
    ax.set_ylim(-0.5, env.n_rows - 0.5)
    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Match matrix indexing
    plt.grid()

    # Arrow mapping: 0:N, 1:S, 2:E, 3:W -> dy, dx
    action_arrows = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            pos = (r, c)
            if pos in env.obstacles:
                ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='black'))
                continue
            if pos == env.terminal_pos:
                ax.text(c, r, 'G', ha='center', va='center', color='green', fontsize=20)
                continue

            state_idx = r * env.n_cols + c
            action = np.argmax(q_table[state_idx])
            dy, dx = action_arrows[action]
            # Quiver takes (x, y, u, v) where u is dx and v is -dy
            ax.quiver(c, r, dx, -dy, color='red', scale=5, headwidth=4)

    plt.title('Learned Policy Map')
    plt.savefig(os.path.join('results', filename))
    plt.show()
    print(f"Saved policy map to results/{filename}")


def run_lr_experiment(filename='learning_rate_comparison.png'):
    """Trains the agent with different learning rates and plots the results."""
    learning_rates = [1e-2, 1e-3, 1e-4]
    results = {}

    for lr in learning_rates:
        print(f"\n--- Training with Learning Rate: {lr} ---")
        # Use fewer episodes for the experiment to save time
        _, rewards_log = train_dqn(learning_rate=lr, num_episodes=150, early_stop_reward=10)
        results[lr] = rewards_log

    plt.figure(figsize=(10, 5))
    for lr, rewards in results.items():
        plt.plot(rewards, label=f'LR = {lr}', alpha=0.8)

    plt.title('Training Performance Comparison by Learning Rate')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join('results', filename))
    plt.show()
    print(f"Saved learning rate comparison to results/{filename}")


def plot_heatmap_with_obstacles(dqn, filename='heatmap.png'):
    """Plots the state-value heatmap with obstacles blacked out."""
    env = GridWorldEnv()
    grid = np.zeros((env.n_rows, env.n_cols))

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            pos = (r, c)
            state = env.encode(pos)
            with torch.no_grad():
                value = dqn(torch.tensor(state).unsqueeze(0).float()).max(1)[0].item()
            grid[r, c] = value

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap='viridis')

    # Add numeric values and black out obstacles
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if (r, c) in env.obstacles:
                ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='black'))
            else:
                ax.text(c, r, f"{grid[r, c]:.1f}", ha='center', va='center', color='white')

    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    plt.title("State-Value Heatmap")
    fig.colorbar(im, ax=ax, label='State Value (Max Q-Value)')
    plt.savefig(os.path.join('results', filename))
    plt.show()
    print(f"Saved heatmap to results/{filename}")


