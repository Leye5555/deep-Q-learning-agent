import torch
import torch.nn as nn
import numpy as np
import os
from .replay_buffer import ReplayBuffer
from .env import GridWorldEnv
from .utils import epsilon_greedy
from config import (
    EPISODES, LEARNING_RATE, GAMMA,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    BATCH_SIZE, BUFFER_SIZE, TARGET_UPDATE
)

CHECKPOINT_DIR = "results/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x


def train_dqn(learning_rate=LEARNING_RATE, num_episodes=EPISODES, early_stop_reward=15):
    env = GridWorldEnv()
    state_dim = env.n_rows * env.n_cols
    action_dim = 4
    dqn = DQN(state_dim, action_dim)
    target_dqn = DQN(state_dim, action_dim)
    target_dqn.load_state_dict(dqn.state_dict())

    # Use the passed learning_rate
    optimizer = torch.optim.Adam(dqn.parameters(), lr=learning_rate)

    buffer = ReplayBuffer(size=BUFFER_SIZE)
    rewards_log = []
    epsilon = EPSILON_START

    # Use the passed num_episodes
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            # ... (rest of the while loop is the same) ...
            with torch.no_grad():
                q_values = dqn(torch.tensor(state).unsqueeze(0))
            action = epsilon_greedy(q_values, epsilon)
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            total_reward += reward

            state = next_state
            steps += 1

            if len(buffer) >= BATCH_SIZE:
                # ... (DQN learning logic is the same) ...
                batch = buffer.sample(BATCH_SIZE)
                state_batch = torch.tensor(batch.state)
                action_batch = torch.tensor(batch.action).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
                next_state_batch = torch.tensor(batch.next_state)
                done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

                q_values = dqn(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    next_q = target_dqn(next_state_batch).max(1)[0].unsqueeze(1)
                    target_q = reward_batch + GAMMA * next_q * (1 - done_batch)
                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # ... (target network update, epsilon decay, logging is the same) ...
        if episode % TARGET_UPDATE == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards_log.append(total_reward)

        if (episode + 1) % 30 == 0:
            last30 = rewards_log[-30:] if len(rewards_log) >= 30 else rewards_log
            print(
                f"LR: {learning_rate} | Ep {episode + 1} | Reward: {total_reward} | Mean(last30): {np.mean(last30):.2f}")

        # Use the passed early_stop_reward
        if len(rewards_log) >= 30 and np.mean(rewards_log[-30:]) > early_stop_reward:
            print(f"Early stopping at LR {learning_rate}: mean reward > {early_stop_reward}")
            break

    return dqn, rewards_log
