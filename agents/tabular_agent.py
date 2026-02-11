"""
Tabular Q-Learning Agent

This replicates standard tabular Q-learning with epsilon-greedy exploration.

Inputs:
Environment: 
    num_states, 
    num_actions, 

Agent:
    alpha=0.1, 
    gamma=1.0,
    epsilon_start=1.0, 
    epsilon_end=0.01, 
    epsilon_decay=0.9997
"""

import numpy as np


class TabularQLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, num_states, num_actions, alpha=0.1, gamma=1.0,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9997):
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # one entry per (state, action) pair - starts at zero
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        """Choose action via epsilon-greedy."""

        # explore randomly with probability epsilon
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # otherwise pick the best known action
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Q-learning update: Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',·) - Q(s,a)]."""

        # if at end of trajectory, there's no future reward. Compute target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        # Compute error and update Q-table
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Clear the Q-table and reset exploration for a fresh run."""

        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.epsilon = self.epsilon_start
