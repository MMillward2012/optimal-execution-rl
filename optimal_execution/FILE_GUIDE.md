# File Guide

This file gives a short description of each file in the `optimal_execution` project.

## calibration

- `calibration/data.py`
  loads raw LOBSTER files, processes the message/book data, and computes the basic calibration inputs such as rates, volatility, book shape, and size distributions

- `calibration/calibrate.py`
  runs the full calibration pipeline for each stock and saves both raw params and RL-smoothed params

## simulator

- `simulator/order_book.py`
  stores the bid and ask sides of the book and implements limit order placements, cancellation, market order executions, and state snapshots

- `simulator/background_order_flow.py`
  simulates the background stream of limit orders, cancellations, and market orders using calibrated parameters

## rl

- `rl/environment.py`
  turns the simulator into an execution environment where the agent sells inventory over time

- `rl/agent.py`
  implements the Double DQN agent

- `rl/agent_safety.py`
  contains extra validation checks and recovery code for NaNs or unstable network updates

- `rl/networks.py`
  defines the Q-network architectures (Double DQN or Duelling)

- `rl/replay_buffer.py`
  stores transitions for training

- `rl/train.py`
  trains the DQN agent using the execution environment

## evaluation

- `evaluation/baselines.py`
  defines benchmark execution policies such as TWAP, passive, aggressive, and AC-style schedules

- `evaluation/plotting.py`
  contains simple plotting helpers for trajectory visualisation

- `evaluation/evaluate_refined.py`
  loads a trained agent, runs evaluation experiments, compares against baselines, and writes summary outputs

## other project files

- `requirements.txt`
  Python dependencies for the project

- `datasets/`
  LOBSTER data and saved parameter files used by calibration and RL

- `runs/`
  saved training runs, checkpoints, and generated plots or summaries

