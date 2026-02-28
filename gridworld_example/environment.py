"""
Creates grid world environment

Open grid with no internal walls. The agent starts at the top left corner
and needs to reach the bottom right (goal) while avoiding the bottom left
(danger zone).

Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right
Rewards: goal +10, danger -10, every step -1
"""

import numpy as np


class GridWorldEnvironment:
    """Square open-grid environment."""

    def __init__(self, rows=3, cols=3, max_steps=10):
        self.rows = rows
        self.cols = cols
        self.num_states = rows * cols
        self.num_actions = 4  # up, down, left, right
        self.max_steps = max_steps

        # special cells
        self.start_state = 0                    # top-left corner
        self.goal_state = self.num_states - 1   # bottom-right corner
        self.danger_state = (rows - 1) * cols   # bottom-left corner

        # reward structure
        self.goal_reward = 10
        self.danger_reward = -10
        self.step_penalty = -1  # small cost for every step to encourage shorter paths

        self.state = None
        self.steps = 0


    def reset(self):
        """Put the agent back at the start."""

        self.state = self.start_state
        self.steps = 0

        return self.state

    def step(self, action):
        """Take one step and return (next_state, reward, done)."""

        next_state = self._get_next_state(self.state, action)
        self.steps += 1

        # check if we hit a terminal cell
        if next_state == self.goal_state:
            reward = self.goal_reward + self.step_penalty
            done = True
        elif next_state == self.danger_state:
            reward = self.danger_reward + self.step_penalty
            done = True
        else:
            reward = self.step_penalty
            done = self.steps >= self.max_steps 

        self.state = next_state
        return next_state, reward, done


    def get_state_as_vector(self, state=None):
        """Use one hot encoding state vector (for vanilla DQN)."""

        if state is None:
            state = self.state

        vec = np.zeros(self.num_states, dtype=np.float32)
        vec[state] = 1.0

        return vec

    def get_state_as_coordinates(self, state=None):
        """
        Normalised (x, y) coordinates in [0, 1].
        This gives the network a sense of spatial structure which helps
        it generalise across positions rather than memorising each one.
        """

        if state is None:
            state = self.state

        row, col = divmod(state, self.cols)
        x = col / (self.cols - 1) if self.cols > 1 else 0.0
        y = row / (self.rows - 1) if self.rows > 1 else 0.0

        return np.array([x, y], dtype=np.float32)


    def _get_next_state(self, state, action):
        """Apply action and ensure it cant escape the grid"""

        row, col = divmod(state, self.cols)

        if action == 0:    # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.rows - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.cols - 1, col + 1)
            
        return row * self.cols + col
