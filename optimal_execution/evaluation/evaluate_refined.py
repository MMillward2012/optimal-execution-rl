import argparse
import json
import sys
from pathlib import Path

import numpy as np
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import baseline policies
from evaluation.baselines import make_baseline_policies
from evaluation.plotting import plot_inventory_trajectories
from rl.agent import DoubleDQNAgent
from rl.agent_safety import network_has_bad_weights
from rl.environment import make_env_from_params


def make_env(config):
    return make_env_from_params(
        ticker=config.get("ticker", "AAPL"),
        total_shares=int(config.get("total_shares", 12000)),
        max_steps=int(config.get("max_steps", 50)),
        steps_per_action=int(config.get("steps_per_action", 20)),
        warmup_steps=int(config.get("warmup_steps", 600)),
        risk_aversion=float(config.get("risk_aversion", 0.0025)),
        urgency_coeff=float(config.get("urgency_coeff", 0.0)),
        urgency_shape=config.get("urgency_shape", "linear"),
        impact_coeff=float(config.get("impact_coeff", 0.01)))


def load_agent(config, checkpoint_path):
    agent = DoubleDQNAgent(
        state_dimensions=int(config.get("state_dim", 6)),
        n_actions=int(config.get("n_actions", 51)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        learning_rate=float(config.get("lr", 1e-4)),
        gamma=float(config.get("gamma", 0.99)),
        tau=float(config.get("tau", 1.0)),
        buffer_size=int(config.get("buffer_size", 50000)),
        batch_size=int(config.get("batch_size", 64)),
        reward_scale=float(config.get("reward_scale", 1.0)),
        epsilon_start=float(config.get("epsilon_start", 1.0)),
        epsilon_end=float(config.get("epsilon_end", 0.01)),
        epsilon_decay_episodes=int(config.get("epsilon_decay_episodes", 4500)),
        target_update_freq=int(config.get("target_update", 1000)))
    
    agent.load(str(checkpoint_path))
    agent.online_net.eval()
    return agent


def make_dqn_policy(agent):
    def policy(env, state, step):
        return int(agent.select_action(state, eval_mode=True))

    return {"name": "DQN", "fn": policy, "exact_shares": False, "agent": agent}


def make_feature_ablation_policies(agent):
    policies = {"dqn": make_dqn_policy(agent)}
    features = [
        ("no_inventory", "No inventory", 0),
        ("no_time", "No time", 1),
        ("no_spread", "No spread", 2),
        ("no_imbalance", "No imbalance", 3),
        ("no_depth", "No depth", 4),
        ("no_impact", "No impact", 5)]

    for key, name, feature_index in features:
        def policy(env, state, step, index=feature_index):
            ablated_state = np.array(state, copy=True)
            ablated_state[index] = 0.0
            return int(agent.select_action(ablated_state, eval_mode=True))

        policies[key] = {"name": name, "fn": policy, "exact_shares": False, "agent": agent}

    return policies


def drift_regimes(values):
    regimes = []
    for value in values:
        name = f"Drift {value:g}"

        # Each regime is applied to a fresh environment before reset, so the
        # same seeds can be compared under different drift
        def apply(env, drift=value):
            env.calibrated_drift = float(drift)

        regimes.append({"name": name, "apply": apply})
    return regimes

# Use the same logic as drift regimes but instead scale depth and rates
def liquidity_regimes(scales):
    regimes = []
    for scale in scales:
        name = f"Liquidity {scale:g}x"

        # Scale both depth and background limit-order flow so liquidity changes consistently
        def apply(env, value=scale):
            env.depth_shape = env.depth_shape * value
            env.avg_depth = env.avg_depth * value
            for rates in env.rates_by_regime.values():
                rates["lambda_limit"] = np.array(rates["lambda_limit"], copy=True) * value

        regimes.append({"name": name, "apply": apply})
    return regimes


def run_episode(config, policy, regime, seed):

    env = make_env(config)
    regime["apply"](env)
    state = env.reset(seed=seed)
    total_reward = 0.0
    inventory = [env.inventory / env.total_shares]
    final_info = {}

    for step in range(env.max_steps):
        # Baselines choose a number of shares directly
        if policy["exact_shares"]:
            shares = int(policy["fn"](env, step))
            state, reward, done, info = env.step(0, exact_shares=shares)
        else:
            # DQN agent has to also take in the state to determine the action
            action = int(policy["fn"](env, state, step))
            state, reward, done, info = env.step(action)

        total_reward += float(reward)
        final_info = info
        inventory.append(info.get("inventory", env.inventory) / env.total_shares)

        if done:
            break

    return {"seed": int(seed),
            "reward": total_reward,
            "implementation_shortfall": float(final_info.get("implementation_shortfall", np.nan)),
            "terminal_filled": float(final_info.get("terminal_filled", 0.0)),
            "inventory": inventory}


def summarise(groups):
    summary = {}

    for label, episodes in groups.items():
        rewards = np.array([episode["reward"] for episode in episodes])
        shortfalls = np.array([episode["implementation_shortfall"] for episode in episodes])
        terminal = np.array([episode["terminal_filled"] for episode in episodes])

        summary[label] = {"episodes": int(len(episodes)),
                          "mean_reward": float(np.mean(rewards)),
                          "std_reward": float(np.std(rewards)),
                          "mean_implementation_shortfall": float(np.mean(shortfalls)),
                          "std_implementation_shortfall": float(np.std(shortfalls)),
                          "mean_terminal_filled": float(np.mean(terminal)),
                          "max_terminal_filled": float(np.max(terminal))}
    return summary


def choose_regular_policies(dqn_policy, params, requested, ac_risk_levels):
    """Used for determining which policies to inclued in the different plots"""
    policies = {"dqn": dqn_policy}
    baseline_policies = make_baseline_policies(params, ac_risk_levels=ac_risk_levels)

    if "all" in requested:
        requested = ["dqn", "twap", "ac", "passive", "aggressive"]

    for key in ("twap", "passive", "aggressive"):
        if key in requested:
            policies[key] = baseline_policies[key]

    if "ac" in requested:
        # AC can produce several schedules, one for each requested risk level.
        for key, policy in baseline_policies.items():
            if key.startswith("ac"):
                policies[key] = policy

    return policies


def run_regular_policies(config, policies, seeds):
    # Take a policy, run it on the normal environment for all seeds, and return the raw trajectories and summary stats
    regular = {"name": "Regular", "apply": lambda env: None}
    groups = {}
    summary = {}

    # Keep the raw trajectories for plotting and the summary stats
    for policy_key, policy in policies.items():
        episodes = [run_episode(config, policy, regular, seed) for seed in seeds]
        groups[policy["name"]] = episodes
        summary[policy_key] = summarise({policy["name"]: episodes})[policy["name"]]

    return groups, summary


def plot_regime_set(config, policy, regimes, seeds, output_path, title):
    # Loop through each regime and run the policy on it for all seeds, then plot the trajectories and return summary stats
    groups = {}
    for regime in regimes:
        groups[regime["name"]] = [run_episode(config, policy, regime, seed) for seed in seeds]

    plot_inventory_trajectories(
        groups,
        output_path,
        title,
        show_examples=False,
        show_std=True)
    
    return summarise(groups)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--checkpoint", default="final_model.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["all"],
        choices=["all", "dqn", "twap", "ac", "passive", "aggressive"],
    )
    parser.add_argument("--ac-risk-levels", nargs="+", type=float, default=[1.0, 10.0, 100.0])
    parser.add_argument("--drift-values", nargs="+", type=float, default=[-0.04, -0.02, 0.0, 0.02, 0.04])
    parser.add_argument("--liquidity-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    return parser.parse_args()


def main():
    # Get command line arguments and load config params
    args = parse_args()
    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text())
    params_path = Path(__file__).resolve().parent.parent / "datasets" / "params" / f"{config.get('ticker', 'AAPL')}_rl_params.json"
    params = json.loads(params_path.read_text())
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    output_dir = args.output_dir or run_dir / "eval_refined"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose seeds, setup the agent and policies
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    agent = load_agent(config, checkpoint_path)
    dqn_policy = make_dqn_policy(agent)
    regular_policies = choose_regular_policies(dqn_policy, params, args.strategies, args.ac_risk_levels)

    # Metadata about which eval was run adn what policies are included
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "seed_end": seeds[-1],
        "regular": {},
        "drift": {},
        "liquidity": {}}

    regular_groups, summary["regular"] = run_regular_policies(config, regular_policies, seeds)

    # Plot inventory trajectories for each regime
    plot_inventory_trajectories(
        {"DQN": regular_groups["DQN"]},
        output_dir / "regular_dqn_examples.png",
        "DQN Regular Inventory Trajectories",
        show_examples=True,
        show_std=False,
        max_examples=50,
    )
    plot_inventory_trajectories(
        regular_groups,
        output_dir / "regular_dqn_baselines.png",
        "Regular Inventory Trajectories",
        show_examples=False,
        show_std=True,
    )

    summary["drift"] = plot_regime_set(
        config,
        dqn_policy,
        drift_regimes(args.drift_values),
        seeds,
        output_dir / "drift_dqn_regimes.png",
        "DQN Inventory Trajectories by Drift Regime",
    )
    summary["liquidity"] = plot_regime_set(
        config,
        dqn_policy,
        liquidity_regimes(args.liquidity_scales),
        seeds,
        output_dir / "liquidity_dqn_regimes.png",
        "DQN Inventory Trajectories by Liquidity Regime",
    )

    feature_ablation_groups, summary["feature_ablation"] = run_regular_policies(config, make_feature_ablation_policies(agent), seeds)
    plot_inventory_trajectories(
        feature_ablation_groups,
        output_dir / "regular_feature_ablation.png",
        "DQN Feature Ablation",
        show_examples=False,
        show_std=True,
    )

    summary["dqn_bad_weights"] = bool(
        network_has_bad_weights(agent.online_net) or network_has_bad_weights(agent.target_net)
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved evaluation to: {output_dir}")
    print(f"Saved summary to:    {summary_path}")


if __name__ == "__main__":
    main()
