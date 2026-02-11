"""
Compare Q-Learning Methods on a 50x50 grid world

Trains four agents and saves results to a single CSV:
  1. Tabular Q-Learning
  2. Vanilla DQN  (no replay, no target)
  3. DQN + Experience Replay
  4. DQN + Replay + Target Network
"""

import numpy as np
import csv
from collections import deque

from environment import GridWorldEnvironment
from agents import TabularQLearningAgent, DQNAgent

# COnfig
GRID_SIZE = 50
MAX_STEPS = 200           # agent gets cut off after this many steps
NUM_EPISODES = 10000

ALPHA_TABULAR = 0.5       # tabular can handle a higher learning rate
ALPHA_DQN = 0.001         # standard Adam learning rate for the neural net agents
GAMMA = 1.0               # no discounting, we care about total reward
EPSILON_START = 1.0       # start fully random
EPSILON_END = 0.01        # always keep a tiny bit of exploration
EPSILON_DECAY = 0.999     # multiplicative decay per episode
WINDOW_SIZE = 100         # moving-average window for calculating average rewards

OUTPUT_CSV = "results.csv"

# Training loop
def train_agent(env, agent, num_episodes, feature_type="tabular"):
    """
    Train an agent and return a list of moving-average rewards.

    feature_type controls how the state is fed to the agent:
      "tabular" -> raw state index (for the Q-table agent)
      "coordinates" -> normalised (x,y) pair (for the DQN agents)
    """

    avg_rewards = []
    reward_window = deque(maxlen=WINDOW_SIZE)

    for episode in range(num_episodes):
        state = env.reset()

        # convert state for DQN agents if needed
        if feature_type == "coordinates":
            state_vec = env.get_state_as_coordinates(state)
        else:
            None

        total_reward = 0
        done = False

        while not done:
            # pick an action
            action = agent.choose_action(state if feature_type == "tabular" else state_vec)

            next_state, reward, done = env.step(action)
            total_reward += reward

            # update the agent
            if feature_type == "tabular":
                agent.update(state, action, reward, next_state, done)
            else:
                next_vec = env.get_state_as_coordinates(next_state)

                agent.update(state_vec, action, reward, next_vec, done)
                state_vec = next_vec

            state = next_state

        # track performance with a rolling average
        reward_window.append(total_reward)
        avg_rewards.append(np.mean(reward_window))

        agent.decay_epsilon()

        if (episode + 1) % 2000 == 0:
            print(f"Episode {episode+1}/{num_episodes}  "
                  f"Avg Reward: {avg_rewards[-1]:.2f}")

    return avg_rewards


# Main loop
def main():
    # create environment
    env = GridWorldEnvironment(rows=GRID_SIZE, cols=GRID_SIZE, max_steps=MAX_STEPS)
    print(f"Environment: {GRID_SIZE}×{GRID_SIZE} grid  ({env.num_states} states)")

    # shared settings for all DQN variants
    # input size is 2 because we use (x, y) coordinate features
    common_dqn_args = dict(
        num_states=2,
        num_actions=env.num_actions,
        alpha=ALPHA_DQN,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=64,
    )

    # each entry: (name, feature_type, agent)
    methods = [
        # tabular
        ("Tabular Q-Learning", "tabular",
         TabularQLearningAgent(
             num_states=env.num_states,
             num_actions=env.num_actions,
             alpha=ALPHA_TABULAR,
             gamma=GAMMA,
             epsilon_start=EPSILON_START,
             epsilon_end=EPSILON_END,
             epsilon_decay=EPSILON_DECAY)),

        # vanilla DQN
        ("Vanilla DQN", "coordinates",
         DQNAgent(**common_dqn_args,
                  use_replay=False, use_target_network=False)),

        # vanilla DQN + replay
        ("DQN + Replay", "coordinates",
         DQNAgent(**common_dqn_args,
                  use_replay=True, use_target_network=False)),

        # vanilla DQN + replay + target (Standard DQN)
        ("DQN + Replay + Target", "coordinates",
         DQNAgent(**common_dqn_args,
                  use_replay=True, use_target_network=True,
                  target_update_freq=100)),
    ]


    results = {}
    for name, feat, agent in methods:
        print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
        results[name] = train_agent(env, agent, NUM_EPISODES, feature_type=feat)

    # save everything to one CSV so plot_results.py can read it later
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode"] + list(results.keys()))
        
        for i in range(NUM_EPISODES):
            writer.writerow([i + 1] + [results[m][i] for m in results])

    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
