import torch
import torch.nn as nn
import torch.nn.functional as F


# A helper to initialise linear layers with Xavier initialisation
def _init_linear_layers(layers):
    for layer in layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


class QNetwork(nn.Module):
    """
    Q-Network for estimating action values.

    Architecture: state -> Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU -> Q-values

    LayerNorm has been added for further stability, but i think now that the NaN issues are resolved
    it might not be necessary.
    """

    def __init__(self, state_dim, n_actions, hidden_dim=64):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

        # Initialise weights using Xavier/Glorot
        _init_linear_layers([self.fc1, self.fc2, self.fc3])

    def forward(self, state):

        activation = F.relu(self.ln1(self.fc1(state)))
        activation = F.relu(self.ln2(self.fc2(activation)))

        q_values = self.fc3(activation)

        return q_values


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.

    Separates state-value V(s) and advantage A(s,a) estimation:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,.))

    Not used any more but it has been tested and is working.
    """

    def __init__(self, state_dim, n_actions, hidden_dim=64):
        super().__init__()

        # Shared feature layer
        self.fc_shared = nn.Linear(state_dim, hidden_dim)
        self.ln_shared = nn.LayerNorm(hidden_dim)

        # Value stream
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)
        self.ln_value = nn.LayerNorm(hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        # Advantage stream
        self.fc_advantage = nn.Linear(hidden_dim, hidden_dim)
        self.ln_advantage = nn.LayerNorm(hidden_dim)
        self.advantage = nn.Linear(hidden_dim, n_actions)

        _init_linear_layers([self.fc_shared, self.fc_value, self.value, self.fc_advantage, self.advantage])

    def forward(self, state):
        x = F.relu(self.ln_shared(self.fc_shared(state)))

        # Value stream
        v = F.relu(self.ln_value(self.fc_value(x)))
        v = self.value(v)

        # Advantage stream
        a = F.relu(self.ln_advantage(self.fc_advantage(x)))
        a = self.advantage(a)

        # Combine: Q = V + (A - mean(A))
        q_values = v + (a - a.mean(dim=1, keepdim=True))
        return q_values
