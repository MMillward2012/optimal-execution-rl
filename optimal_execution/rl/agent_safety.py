"""Small numerical safety helpers for the DQN agent.

I have written this in the AI declaration, but this file was created by Claude Opus 4.6, to assist with 
debugging and recovering from rare NaN/Inf failures during training. It does not affect the learning 
algorithm, but it protects a long training run from being ruined by a single bad batch, NaN weights, 
or a broken target network.

After extensive debugging, the problems have been resolved now by switching to AdamW, 
but I have kept these functions here in case they are needed again in the future.
"""

import torch


def reinitialise_linear_layers(network):
    for module in network.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)


def tensor_has_bad_values(tensor):
    """Return True when a tensor contains NaN or Inf values."""
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def batch_has_bad_values(*tensors):
    """Return True when any tensor in a sampled batch contains NaN or Inf."""
    for tensor in tensors:
        if tensor_has_bad_values(tensor):
            return True
    return False


def network_has_bad_weights(network):
    """Return True when any network parameter contains NaN or Inf."""
    for parameter in network.parameters():
        if tensor_has_bad_values(parameter.data):
            return True
    return False


def safe_double_dqn_next_q(
    online_net,
    target_net,
    next_states,
    training_step,
    nan_recovery_count,
    logger,
):
    """Compute the Double DQN next-state value, repairing rare NaN failures.

    Returns ``(next_q, nan_recovery_count)``.  ``next_q`` is ``None`` only when
    both networks were unusable and the update should be skipped.
    """
    online_output = online_net(next_states)
    next_actions = online_output.argmax(dim=1)
    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

    if not tensor_has_bad_values(next_q):
        return next_q, nan_recovery_count

    if tensor_has_bad_values(online_output):
        nan_recovery_count += 1
        logger.error(
            "Both networks produced NaN at step %d; reinitialising weights (recovery #%d)",
            training_step,
            nan_recovery_count,
        )
        reinitialise_linear_layers(online_net)
        target_net.load_state_dict(online_net.state_dict())
        return None, nan_recovery_count

    logger.warning(
        "Target network produced NaN at step %d; restoring it from the online network",
        training_step,
    )
    target_net.load_state_dict(online_net.state_dict())
    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
    return next_q, nan_recovery_count


def log_bad_loss(
    logger,
    training_step,
    loss,
    current_q,
    target_q,
    rewards,
    next_q,
):
    """Log and return True if a loss is NaN or Inf."""
    if not tensor_has_bad_values(loss):
        return False

    logger.warning(
        "NaN/Inf loss at step %d | "
        "current_q [%.2f, %.2f] | target_q [%.2f, %.2f] | "
        "rewards [%.2f, %.2f] | next_q [%.2f, %.2f]",
        training_step,
        current_q.min().item(),
        current_q.max().item(),
        target_q.min().item(),
        target_q.max().item(),
        rewards.min().item(),
        rewards.max().item(),
        next_q.min().item(),
        next_q.max().item(),
    )
    return True


def update_target_network(
    online_net,
    target_net,
    tau,
    logger,
):
    """Copy the online network into the target network.

    ``tau >= 1`` means a full copy.  ``tau < 1`` means a soft moving average.
    """
    if network_has_bad_weights(online_net):
        logger.warning("Skipping target update because online network contains NaN/Inf")
        return

    if tau >= 1.0:
        target_net.load_state_dict(online_net.state_dict())
        return

    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
