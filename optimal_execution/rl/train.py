import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import json
import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.agent import DoubleDQNAgent
from rl.environment import make_env_from_params


# Training parameters.  Most remain fixed across experiments,
# only risk aversion and urgency need to be changed to see different behaviours.
RUN_LABEL = "baseline"
RUN_NAME = None  
OUTPUT_DIR = "runs"
EPISODES_OVERRIDE = None

# 7500 episodes used for the dis results, but 5000 is enough to see good results so i use
# that for debugging.
EPISODES = 5000
EVAL_FREQ = 500
EVAL_EPISODES = 20
FINAL_EVAL_EPISODES = 100
SAVE_FREQ = 500
PROGRESS_FREQ = 50

TICKER = "AAPL"
TOTAL_SHARES = 12_000
MAX_STEPS = 50
STEPS_PER_ACTION = 20
WARMUP_STEPS = 600
IMPACT_COEFF = 0.3

RISK_AVERSION = 0.0015
URGENCY_COEFF = 10.0
URGENCY_SHAPE = "quadratic"

HIDDEN_DIM = 64
LEARNING_RATE = 1e-4
GAMMA = 0.99
TAU = 1.0
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
REWARD_SCALE = 1000.0
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_EPISODES = 4500
TARGET_UPDATE = 1_000

if EPISODES_OVERRIDE != None:
    ACTIVE_EPISODES = EPISODES_OVERRIDE 
else:
    ACTIVE_EPISODES = EPISODES

ACTIVE_MAX_STEPS = MAX_STEPS
ACTIVE_EVAL_FREQ = EVAL_FREQ
ACTIVE_EVAL_EPISODES = EVAL_EPISODES
ACTIVE_FINAL_EVAL_EPISODES = FINAL_EVAL_EPISODES
ACTIVE_SAVE_FREQ = SAVE_FREQ
ACTIVE_PROGRESS_FREQ = PROGRESS_FREQ
EVAL_SEED_OFFSET = 10_000

BASE_FEATURES = ["inventory_remaining", "time_remaining", "spread_norm", "book_imbalance", "relative_depth", "recent_impact_proxy"]


def make_env():
    """Create the environment"""
    return make_env_from_params(
        ticker=TICKER,
        total_shares=TOTAL_SHARES,
        max_steps=ACTIVE_MAX_STEPS,
        steps_per_action=STEPS_PER_ACTION,
        warmup_steps=WARMUP_STEPS,
        risk_aversion=RISK_AVERSION,
        urgency_coeff=URGENCY_COEFF,
        urgency_shape=URGENCY_SHAPE,
        impact_coeff=IMPACT_COEFF)


def make_agent(env, logger):
    """Create the DQN agent"""
    return DoubleDQNAgent(
        state_dimensions=env.state_dim,
        n_actions=env.N_ACTIONS,
        hidden_dim=HIDDEN_DIM,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        reward_scale=REWARD_SCALE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        target_update_freq=TARGET_UPDATE,
        logger=logger)


def evaluate(env, n_episodes, agent):
    """Evaluate the greedy agent."""
    rewards, shortfalls, risks = [], [], []

    # Use fixed seeds for repeatability. A change in results should
    # then come from the policy, not from getting lucky episodes
    for seed in range(EVAL_SEED_OFFSET, EVAL_SEED_OFFSET + n_episodes):
        state = env.reset(seed=seed)
        total_reward = 0.0

        while True:
            # eval_mode uses the greedy policy
            # Pick an action, do the step, and compute rewards
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                rewards.append(total_reward)
                shortfalls.append(info["implementation_shortfall"])
                risks.append(info["risk_penalty"])
                break

    return {"mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_is": float(np.mean(shortfalls)),
            "std_is": float(np.std(shortfalls)),
            "mean_risk_penalty": float(np.mean(risks))}


def train_episode(agent, env, seed):
    """Run one episode and train from replay after each transition"""
    # Observe the state, choose action, step environment, store the
    # transition, then let the agent train from replay
    state = env.reset(seed=seed)
    reward_sum = 0.0
    losses = []

    while True:
        # Pick an action and add the transition to the buffer
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)

        # Compute the loss and perform a gradient step
        loss = agent.train_step()
        if loss != None:
            losses.append(loss)

        # Compute reward and update state, then if episode is done update epsilon
        reward_sum += reward
        state = next_state
        if done:
            agent.update_epsilon()
            mean_loss = float(np.mean(losses)) if losses else None
            return reward_sum, mean_loss, info


def print_header(env, eval_env, run_dir, milestones):
    """Header for training start summarising settings"""
    print("=" * 60)
    print("DOUBLE DQN TRAINING")
    print("=" * 60)
    print(f"Run: {run_dir}")
    print(f"Label: {RUN_LABEL} | episodes: {ACTIVE_EPISODES} | device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"State: {env.state_dim} ({', '.join(BASE_FEATURES)})")
    print(f"Risk={RISK_AVERSION:g} | urgency={URGENCY_SHAPE}:{URGENCY_COEFF:g} | "
          f"impact={IMPACT_COEFF:g} | drift={eval_env.calibrated_drift:.6f}")
    print(f"Steps/action: {STEPS_PER_ACTION} | warmup: {WARMUP_STEPS} | LR decays at {milestones or 'none'}")
    print("=" * 60)


def save_run_config(env, run_dir):
    """Save the settings needed by evaluation scripts to a json file"""
    saved = {
        "preset": RUN_LABEL,
        "ticker": TICKER,
        "total_shares": TOTAL_SHARES,
        "max_steps": ACTIVE_MAX_STEPS,
        "steps_per_action": STEPS_PER_ACTION,
        "warmup_steps": WARMUP_STEPS,
        "risk_aversion": RISK_AVERSION,
        "urgency_coeff": URGENCY_COEFF,
        "urgency_shape": URGENCY_SHAPE,
        "impact_coeff": IMPACT_COEFF,
        "episodes": ACTIVE_EPISODES,
        "eval_freq": ACTIVE_EVAL_FREQ,
        "eval_episodes": ACTIVE_EVAL_EPISODES,
        "final_eval_episodes": ACTIVE_FINAL_EVAL_EPISODES,
        "save_freq": ACTIVE_SAVE_FREQ,
        "progress_freq": ACTIVE_PROGRESS_FREQ,
        "hidden_dim": HIDDEN_DIM,
        "lr": LEARNING_RATE,
        "gamma": GAMMA,
        "tau": TAU,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "reward_scale": REWARD_SCALE,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_episodes": EPSILON_DECAY_EPISODES,
        "target_update": TARGET_UPDATE,
        "state_dim": env.state_dim,
        "observation_features": list(BASE_FEATURES),
        "simulator_profile": "simple_background_order_flow",
        "parameter_profile": env.params["metadata"].get("parameter_profile", "raw_calibrated"),
        "regimes": list(env.rates_by_regime.keys())
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(saved, f, indent=2)


def make_logger(run_dir):
    """Create a logger for recording agent warnings and progress messages"""
    logger = logging.getLogger("dqn_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(run_dir / "training.log")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def log_training_scalars(writer, episode, reward, loss, info, agent, scheduler):
    """Log training metrics to TensorBoard for monitoring"""
    writer.add_scalar("train/episode_reward", reward, episode)
    writer.add_scalar("train/epsilon", agent.epsilon, episode)
    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], episode)
    writer.add_scalar("train/buffer_size", agent.buffer.size, episode)
    writer.add_scalar("train/is", info["implementation_shortfall"], episode)
    writer.add_scalar("train/risk_penalty", info["risk_penalty"], episode)
    if loss is not None:
        writer.add_scalar("train/loss", loss, episode)


def log_eval_scalars(writer, episode, stats):
    """Log evaluation metrics to TensorBoard for monitoring"""
    writer.add_scalar("eval/mean_reward", stats["mean_reward"], episode)
    writer.add_scalar("eval/std_reward", stats["std_reward"], episode)
    writer.add_scalar("eval/mean_is", stats["mean_is"], episode)
    writer.add_scalar("eval/mean_risk_penalty", stats["mean_risk_penalty"], episode)

def format_progress_message(episode, recent_rewards, recent_losses, agent, speed):
    return (f"Episode {episode:5d} | Reward: {np.mean(recent_rewards):8.1f} | "
            f"Loss: {(np.mean(recent_losses) if recent_losses else 0.0):.4f} | "
            f"Epsilon: {agent.epsilon:.3f} | Buffer: {agent.buffer.size:6d} | Speed: {speed:.1f} ep/s")


def main():
    # Set up directories
    run_dir_name = RUN_NAME or RUN_LABEL
    run_dir = Path(OUTPUT_DIR) / run_dir_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Make an environment for training and for eval
    train_env = make_env()
    eval_env = make_env()
    save_run_config(train_env, run_dir)

    # Create the agent, logger, learning rate scheduler and TensorBoard writer
    logger = make_logger(run_dir)
    agent = make_agent(train_env, logger)
    milestones = sorted({int(ACTIVE_EPISODES * 0.6), int(ACTIVE_EPISODES * 0.8)} - {0})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(agent.optimiser, milestones=milestones, gamma=0.5)
    tb_writer = SummaryWriter(run_dir / "tensorboard")

    # Print the header now we are ready to start training
    print_header(train_env, eval_env, run_dir, milestones)
    start = time.time()
    recent_rewards, recent_losses = deque(maxlen=100), deque(maxlen=100)

    try:
        for episode in range(1, ACTIVE_EPISODES + 1):
            # Run a full training episode
            reward, loss, info = train_episode(agent, train_env, seed=episode - 1)
            recent_rewards.append(reward)

            # Only update the learning rate scheduler after a valid training step
            if loss != None:
                scheduler.step()
                recent_losses.append(loss)

            # Save training metrics to TensorBoard for monitoring
            log_training_scalars(tb_writer, episode, reward, loss, info, agent, scheduler)

            # Print a progress message after ACTIVE_PROGRESS_FREQ episodes
            if episode % ACTIVE_PROGRESS_FREQ == 0:
                speed = episode / max(time.time() - start, 1e-9)
                msg = format_progress_message(episode, recent_rewards, recent_losses, agent, speed)
                print(msg)
                logger.info(msg)

            # Run evaluation episodes and log results after ACTIVE_EVAL_FREQ episodes
            if episode % ACTIVE_EVAL_FREQ == 0:
                stats = evaluate(eval_env, ACTIVE_EVAL_EPISODES, agent)
                log_eval_scalars(tb_writer, episode, stats)
                print(f"\nEval episode {episode}"
                      f"\n  reward: {stats['mean_reward']:.1f} +/- {stats['std_reward']:.1f}"
                      f"\n  IS:     ${stats['mean_is']:.2f}")

            # Save a checkpoint after ACTIVE_SAVE_FREQ episodes
            if episode % ACTIVE_SAVE_FREQ == 0:
                agent.save(checkpoint_dir / f"checkpoint_{episode}.pt")

        # Once training is complete, save the final model and run a final eval
        agent.save(checkpoint_dir / "final_model.pt")
        final = evaluate(eval_env, ACTIVE_FINAL_EVAL_EPISODES, agent)
        minutes = (time.time() - start) / 60
        print("\n" + "=" * 60 + "\n"
              "Final Results\n"
              f"  Agent reward: {final['mean_reward']:.1f} +/- {final['std_reward']:.1f}\n"
              f"  Agent IS:     ${final['mean_is']:.2f} +/- ${final['std_is']:.2f}\n"
              f"  Time:         {minutes:.1f} minutes\n"
              f"  Saved to:     {run_dir}")
    finally:
        # Close the writer even if training is interrupted
        tb_writer.close()


if __name__ == "__main__":
    main()