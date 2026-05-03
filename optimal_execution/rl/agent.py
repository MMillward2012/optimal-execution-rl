import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import logging


# Note: Claude Code was responsible for generating these safety measures imported from agent_safety.py,
# The rest of the code has been completed by myself and I have kept agent_safety.py as a separate file.
# The problem i kept encountering was that the main network would produce NaN q-values at around 4500/5000
# training steps, corrupting the target network and wasting a lot of time before I realised what was going on
from .agent_safety import (
    batch_has_bad_values,
    log_bad_loss,
    network_has_bad_weights,
    safe_double_dqn_next_q,
    update_target_network,
)

from .networks import QNetwork
from .replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    def __init__(
        self,
        state_dimensions,
        n_actions,
        hidden_dim=64,
        learning_rate=1e-4,
        gamma=0.99,
        tau=1.0,  # tau=1.0 for hard update
        buffer_size=50_000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=4_000,
        target_update_freq=1000,
        reward_scale=1.0,
        logger=None):

        # Logger used for debugging and typical initialisation
        self.logger = logger or logging.getLogger(__name__)
        self.state_dim = state_dimensions
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau  
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.episodes_done = 0
        self.epsilon = epsilon_start
        self.reward_scale = float(reward_scale)
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")

        # was going to add MPS support but last time i tried it didnt work - sticking with Cuda and CPU for now
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialise the Q-networks
        self.online_net = QNetwork(state_dimensions, n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dimensions, n_actions, hidden_dim).to(self.device)
        # Give the target network the same initial weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # only use target network for inference

        # Tried regular Adam but q-values were diverging, AdamW resolved the problem
        self.optimiser = optim.AdamW(self.online_net.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Set up replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dimensions)

        # Counters and NaN stuff for debugging
        self.total_steps = 0
        self.training_steps = 0
        self.nan_recovery_count = 0  # Track how many times weve had to reinitialise


    def convert_batch_to_tensors(self, states, actions, rewards, next_states, done):
        """convert the features to tensors"""
        return (
            torch.as_tensor(states, dtype=torch.float32, device=self.device),
            torch.as_tensor(actions, dtype=torch.long, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(next_states, dtype=torch.float32, device=self.device),
            torch.as_tensor(done, dtype=torch.bool, device=self.device)
            )


    def select_action(self, state, eval_mode=False):
        """Use epsilon greedy to pick an action"""

        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)

        # Otherwise pick the action with the highest Q value for this state.
        with torch.no_grad():
            s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qs = self.online_net(s_tensor)
            return qs.argmax(dim=1).item()


    def store_transition(self, state, action, reward, next_state, done):
        """Add the transition to the replay buffer"""

        self.buffer.push(state, action, reward / self.reward_scale, next_state, done)
        self.total_steps += 1


    def update_epsilon(self):
        """update epsilon after an episode finishes"""

        self.episodes_done += 1

        progress = min(1.0, self.episodes_done / self.epsilon_decay_episodes)

        self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress


    def train_step(self):
        """one learning update, if the replay buffer has enough data"""

        # check if we have enough samples in the replay buffer to start training
        if not self.buffer.is_ready(self.batch_size):
            return None

        # Sample batch from buffer
        states, actions, rewards, next_states, done = self.buffer.sample(self.batch_size)

        # Convert to tensors for pytorch
        states, actions, rewards, next_states, done = self.convert_batch_to_tensors(states, actions, rewards, next_states, done)

        # Use the helper to check if the sampled data contains NaN or Inf values.
        if batch_has_bad_values(states, next_states, rewards):
            self.logger.warning(
                'NaN in sampled batch at step %d — skipping',
                self.training_steps,
            )
            return None

        # Get the Q value for the action the agent took
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)


        # Double DQN target.
        with torch.no_grad():
            # Get the action with the highest Q value for the next state
            # Use the helper for this because we dont want to corrupt training
            next_q, self.nan_recovery_count = safe_double_dqn_next_q(self.online_net, self.target_net, next_states, self.training_steps, self.nan_recovery_count, self.logger)
            if next_q is None:
                return None

            # Compute TD target or if the episode is done then just reward
            target_q = rewards + self.gamma * next_q * (done == False)


        # Huber loss is definitely more stable as it prevents the loss exploding,
        # but MSE still works and matches the claims in my dissertation. I would 
        # probably recommend Huber for anyone implementing this themselves
        # loss = nn.MSELoss()(current_q, target_q)
        loss = nn.HuberLoss(delta=1.0)(current_q, target_q)

        # make sure the loss and q values are not NaN or inf before doing backwards pass
        if log_bad_loss(self.logger, self.training_steps, loss, current_q, target_q, rewards, next_q):
            return None


        # Normal NN update
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimiser.step()


        # If the update broke it, go back to the target network.
        if network_has_bad_weights(self.online_net):
            self.logger.warning(
                'NaN in online network weights after step %d — '
                'restoring from target network',
                self.training_steps,
            )
            self.online_net.load_state_dict(self.target_net.state_dict())
            return None

        self.training_steps += 1

        # Update target network every target_update_freq steps
        if self.training_steps % self.target_update_freq == 0:
            update_target_network(self.online_net, self.target_net, self.tau, self.logger)

        return loss.item()

    def save(self, path):
        """save the model"""
        # Create a checkpoint dict to resume training
        checkpoint = {'online_net': self.online_net.state_dict(),
                    'target_net': self.target_net.state_dict(),
                    'optimiser': self.optimiser.state_dict(),
                    'total_steps': self.total_steps,
                    'training_steps': self.training_steps,
                    'episodes_done': self.episodes_done,
                    'epsilon': self.epsilon,
                    'config': {'state_dim': self.state_dim,
                                'n_actions': self.n_actions,
                                'gamma': self.gamma,
                                'tau': self.tau,
                                'batch_size': self.batch_size,
                                'reward_scale': self.reward_scale,
                                'epsilon_start': self.epsilon_start,
                                'epsilon_end': self.epsilon_end,
                                'epsilon_decay_episodes': self.epsilon_decay_episodes}}
        
        # Ensure the directory exists before saving
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load(self, path):
        """load the model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore the model and training state
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimiser.load_state_dict(checkpoint['optimiser'])
        self.total_steps = checkpoint['total_steps']
        self.training_steps = checkpoint['training_steps']
        self.episodes_done = checkpoint.get('episodes_done', 0)
        self.epsilon = checkpoint['epsilon']
        self.reward_scale = checkpoint.get('config', {}).get('reward_scale', self.reward_scale)

        return None
    
