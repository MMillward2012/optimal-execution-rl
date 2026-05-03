import numpy as np

AC_TEMPORARY_IMPACT = 1e-6
AGGRESSIVE_STEPS = 3
PASSIVE_STEPS = 3

def twap_policy(env, step):
    if env.inventory <= 0:
        return 0
    
    target_next = env.total_shares * (1.0 - (step + 1) / env.max_steps)

    return max(0, int(round(env.inventory - target_next)))


def aggressive_policy(env, step):
    if env.inventory <= 0:
        return 0
    
    if step < AGGRESSIVE_STEPS:
        steps_left = AGGRESSIVE_STEPS - step
        return int(np.ceil(env.inventory / steps_left))
    
    return 0


def passive_policy(env, step):
    if env.inventory <= 0:
        return 0
    
    first_passive_step = max(0, env.max_steps - PASSIVE_STEPS)

    if step >= first_passive_step:
        steps_left = env.max_steps - step
        return int(np.ceil(env.inventory / steps_left))
    
    return 0


def ac_inventory(total_shares, max_steps, params, risk_aversion):
    # Compute parameters for the AC formula
    sigma = float(np.nanmean(params["volatility"]["sigma_k"]) * np.sqrt(max_steps))
    kappa = np.sqrt(max(0.0, risk_aversion) * sigma * sigma / AC_TEMPORARY_IMPACT)
    times = np.arange(max_steps + 1, dtype=float)

    # To avoid 0/0 problems, default to twap because the AC formula converges to TWAP as kappa -> 0
    if kappa <= 1e-12:
        return total_shares * (1.0 - times / max_steps)

    return total_shares * np.sinh(kappa * (max_steps - times)) / np.sinh(kappa * max_steps)


def make_ac_policy(params, risk_aversion):
    def policy(env, step):
        if env.inventory <= 0 or step >= env.max_steps:
            return 0
        
        inventory = ac_inventory(env.total_shares, env.max_steps, params, risk_aversion)
        target_inventory = max(0, int(round(inventory[step + 1])))
        
        return max(0, env.inventory - target_inventory)

    return policy


def make_baseline_policies(params, ac_risk_levels):
    policies = {"twap": {"name": "TWAP", "fn": twap_policy, "exact_shares": True},
                "passive": {"name": "Passive", "fn": passive_policy, "exact_shares": True},
                "aggressive": {"name": "Aggressive", "fn": aggressive_policy, "exact_shares": True}}
    
    for risk in ac_risk_levels:
        key = f"ac{risk:g}"
        policies[key] = {"name": f"AC {risk:g}", "fn": make_ac_policy(params, risk), "exact_shares": True}
        
    return policies
