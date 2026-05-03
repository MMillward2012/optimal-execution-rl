import numpy as np

class ReplayBuffer:
    """
    Parameters:
    - capacity is the size of the buffer
    - state_dim pretty self explanatoru
    """
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.state_dim = state_dim

        self.buffer = []

        self.position = 0 
        self.size = 0 

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer. Rejects entries with NaN/Inf."""

        if np.isnan(state).any() or np.isnan(next_state).any() or np.isnan(reward) or np.isinf(reward):
            return

        transition = (state, action, reward, next_state, done)

        if self.size < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity
        self.size = len(self.buffer)

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def is_ready(self, batch_size):
        """Check if buffer has enough samples to start training.

        Warmup requires at least 1000 samples (~20 episodes) before the first
        gradient update, so mini-batches have enough diversity and samples are
        not reused many times in the first few episodes.
        """
        warmup = max(batch_size, 1000)
        return self.size >= warmup
