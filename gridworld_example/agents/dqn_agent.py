"""
Deep Q-Network (DQN) Agent

This file includes a DQN agent with optional toggleable
experience replay and target network. Can toggle features 
by adjusting use_replay=False, use_target_network=False.
Uses Adam optimiser and MSE loss.

Other input parameters include:

Environment:
    num_states, 
    num_actions, 

Agent:  
    alpha=0.001, 
    gamma=1.0,
    epsilon_start=1.0, 
    epsilon_end=0.01, 
    epsilon_decay=0.9997,

Network:
    hidden_size=64, 

Replay and target network:
    replay_capacity=10000, 
    batch_size=32, 
    target_update_freq=10
"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Simple feedforward net using ReLU: input -> 64 -> 64 -> num_actions."""

    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # raw Q-values, no activation


class ReplayBuffer:
    """Fixed-size replay buffer that overwrites oldest experiences first."""

    def __init__(self, capacity=10000):
        # Works like a list but with a max size. When full, adding a new item 
        # drops the oldest one.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Draw a random mini-batch for training."""

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with configurable experience replay and target network.

    By toggling use_replay and use_target_network we get different variants:
      - Vanilla DQN:            both False
      - DQN + Replay:           use_replay=True
      - DQN + Replay + Target:  both True
    """

    def __init__(self, num_states, num_actions, alpha=0.001, gamma=1.0,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9997,
                 hidden_size=64, use_replay=False, use_target_network=False,
                 replay_capacity=10000, batch_size=32, target_update_freq=100):
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_replay = use_replay
        self.use_target_network = use_target_network
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_size = hidden_size
        self.alpha = alpha

        # use GPU if available, otherwise CPU is fine for this grid size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # main Q-network that we actually update
        self.q_network = QNetwork(num_states, num_actions, hidden_size).to(self.device)
        self.optimiser = optim.Adam(self.q_network.parameters(), lr=alpha)

        # target network is a frozen copy - only synced every N steps
        if use_target_network:
            self.target_network = QNetwork(num_states, num_actions, hidden_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()  # never call .train() on this
        else:
            self.target_network = None

        # replay buffer stores past transitions for mini-batch training
        self.replay_buffer = ReplayBuffer(replay_capacity) if use_replay else None

        self.step_count = 0

    def choose_action(self, state_vector):
        """Epsilon-greedy action selection."""

        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # pick greedy action from Q-network
        with torch.no_grad():
            t = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)

            return self.q_network(t).argmax().item()

    def update(self, state, action, reward, next_state, done):
        """Store transition and train (single-step or mini-batch)."""

        self.step_count += 1

        if self.use_replay:
            # add to buffer and sample a random batch to break correlations
            self.replay_buffer.push(state, action, reward, next_state, done)

            if len(self.replay_buffer) < self.batch_size:
                return  # wait until we have enough samples
            
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            self._train_batch(states, actions, rewards, next_states, dones)
        else:
            # no replay - just train on the single most recent transition
            self._train_batch(
                np.array([state]), np.array([action]), np.array([reward]),
                np.array([next_state]), np.array([done]),
            )

        # periodically copy weights to the target network
        if self.use_target_network and self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _train_batch(self, states, actions, rewards, next_states, dones):
        """Run one gradient step on a batch of transitions."""

        # Convert to tensors and move to correct device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # get Q-values for the actions we actually took
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # compute TD targets, use target net if we have one, otherwise use main net
        with torch.no_grad():
            net = self.target_network if self.use_target_network else self.q_network
            next_q = net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # standard MSE loss between predicted and target Q-values, then backprop and 
        # gradient descent
        loss = nn.MSELoss()(current_q, target_q)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Re-initialise everything for a fresh training run."""

        self.q_network = QNetwork(self.num_states, self.num_actions, self.hidden_size).to(self.device)
        self.optimiser = optim.Adam(self.q_network.parameters(), lr=self.alpha)

        if self.use_target_network:
            self.target_network = QNetwork(self.num_states, self.num_actions, self.hidden_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.use_replay:
            self.replay_buffer = ReplayBuffer()
            
        self.epsilon = self.epsilon_start
        self.step_count = 0
