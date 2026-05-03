import numpy as np
import json
from datetime import datetime
from pathlib import Path

from calibration.data import (
    BOOK_LEVELS,
    BUCKET_SECONDS,
    DEFAULT_DATE,
    DEFAULT_STOCKS,
    N_BUCKETS,
    N_LEVELS,
    RAW_DIR,
    ROOT,
    TICK_SIZE,
    estimate_book_shape,
    estimate_rates,
    estimate_size_distribution,
    estimate_volatility,
    load_raw,
    process)


PARAMS_DIR = ROOT / "datasets" / "params"
RUN_STOCKS = DEFAULT_STOCKS
RUN_OUTPUT_DIR = PARAMS_DIR
RUN_DRIFT_OFFSET = 0.0


def build_params(stock, date, messages, book, drift_offset=0.0):
    # Creates a dictionary of all the calibrated parameters for a given stock and date
    rates = estimate_rates(messages)
    shape = estimate_book_shape(book, BOOK_LEVELS)
    volatility = estimate_volatility(book).tolist()
    size_distribution = estimate_size_distribution(messages)
    arrival_rates = {}
    for key, value in rates.items():
        arrival_rates[key] = value.tolist()

    metadata = {"stock": stock,
                "dates": [date],
                "n_buckets": N_BUCKETS,
                "bucket_duration_sec": BUCKET_SECONDS,
                "n_levels": N_LEVELS,
                "book_levels": BOOK_LEVELS,
                "tick_size": TICK_SIZE,
                "opening_mid": float(book["mid_price"].iloc[0]),
                "calibrated_at": datetime.now().isoformat(),
                "drift_offset": drift_offset}
    
    book_shape = {"depth": shape["depth"].tolist(), "spread_by_bucket": shape["spread_by_bucket"].tolist()}

    return {"metadata": metadata,
        "volatility": {"sigma_k": volatility},
        "arrival_rates": arrival_rates,
        "size_distribution": size_distribution,
        "book_shape": book_shape}


def build_rl_params(params):
    # Basically just do the same except convert into the 3 regimes used for the RL sim
    # Also apply corrections to remove side imbalance and average out the spread
    # Kinda long but includes all the necessary parameters for the simulator along with transformations.
    # This is so the simulator doesnt just reproduce the raw data but instead has smoothed params
    limit = np.array(params["arrival_rates"]["lambda_limit"])
    cancel = np.array(params["arrival_rates"]["lambda_cancel"])
    market = np.array(params["arrival_rates"]["lambda_mo"])
    spread = np.array(params["book_shape"]["spread_by_bucket"])
    sigma = np.array(params["volatility"]["sigma_k"])
    neutral_spread = float(np.nanmean(spread))

    rl_params = {"metadata": dict(params["metadata"]),
                "volatility": dict(params["volatility"]),
                "arrival_rates": dict(params["arrival_rates"]),
                "size_distribution": dict(params["size_distribution"]),
                "book_shape": dict(params["book_shape"])}
    
    rl_params["metadata"]["drift_offset"] = 0.0
    rl_params["metadata"]["neutral_spread"] = neutral_spread

    rl_limit = np.zeros_like(limit)
    rl_cancel = np.zeros_like(cancel)
    rl_market = np.zeros_like(market)
    rl_spread = np.full_like(spread, neutral_spread)
    rl_sigma = np.zeros_like(sigma)
    regime_by_bucket = [None] * N_BUCKETS
    regimes = {}
    calibrated_regimes = {}

    # Use three fixed intraday regimes for the RL environment.
    regime_groups = {"open": list(range(10)), 
                     "midday": list(range(10, N_BUCKETS - 10)), 
                     "close": list(range(N_BUCKETS - 10, N_BUCKETS))}

    # Go through and average the parameters across the regimes
    for name, buckets in regime_groups.items():
        bucket_indices = np.array(buckets)

        limit_in_regime = limit[bucket_indices]
        cancel_in_regime = cancel[bucket_indices]
        market_in_regime = market[bucket_indices]
        spread_in_regime = spread[bucket_indices]
        sigma_in_regime = sigma[bucket_indices]

        raw_limit = np.mean(limit_in_regime, axis=0)
        raw_cancel = np.mean(cancel_in_regime, axis=0)
        raw_market = np.mean(market_in_regime, axis=0)
        raw_spread = float(np.mean(spread_in_regime))
        raw_sigma = float(np.mean(sigma_in_regime))

        # Remove side imbalance but keep the same total activity.
        neutral_limit = remove_side_imbalance(raw_limit)
        neutral_cancel = remove_side_imbalance(raw_cancel)
        neutral_market = remove_side_imbalance(raw_market)
        rl_limit[bucket_indices] = neutral_limit
        rl_cancel[bucket_indices] = neutral_cancel
        rl_market[bucket_indices] = neutral_market
        rl_sigma[bucket_indices] = raw_sigma

        # Update the bucket name
        for bucket in buckets:
            regime_by_bucket[bucket] = name

        regimes[name] = {"buckets": buckets,
            "lambda_limit": neutral_limit.tolist(),
            "lambda_cancel": neutral_cancel.tolist(),
            "lambda_mo": neutral_market.tolist(),
            "spread": neutral_spread,
            "spread_ticks": spread_to_side_offset_ticks(neutral_spread),
            "sigma": raw_sigma}
        
        calibrated_regimes[name] = {"buckets": buckets,
            "lambda_limit": raw_limit.tolist(),
            "lambda_cancel": raw_cancel.tolist(),
            "lambda_mo": raw_market.tolist(),
            "spread": raw_spread,
            "spread_ticks": spread_to_side_offset_ticks(raw_spread),
            "sigma": raw_sigma}

    # Replace the raw bucket-by-bucket arrays with the smoothed RL ones.
    rl_params["arrival_rates"]["lambda_limit"] = rl_limit.tolist()
    rl_params["arrival_rates"]["lambda_cancel"] = rl_cancel.tolist()
    rl_params["arrival_rates"]["lambda_mo"] = rl_market.tolist()
    rl_params["book_shape"]["spread_by_bucket"] = rl_spread.tolist()
    rl_params["volatility"]["sigma_k"] = rl_sigma.tolist()
    rl_params["regimes"] = regimes
    rl_params["calibrated_regimes"] = calibrated_regimes
    rl_params["regime_by_bucket"] = regime_by_bucket
    return rl_params


# A few helpers to make sure that the calibrated params match up with how the simulator uses them
def spread_to_side_offset_ticks(spread):
    # Convert the full spread into the per-side tick offset used by the simulator.
    return max(0.0, spread / TICK_SIZE / 2 - 1)


def remove_side_imbalance(rates):
    # Make bid and ask activity equal while keeping the same total event rate.
    rates = np.array(rates)
    if rates.ndim == 1:
        return np.array([rates.sum() / 2, rates.sum() / 2])

    neutral = np.zeros_like(rates)
    neutral[:, 0] = rates.sum(axis=1) / 2
    neutral[:, 1] = rates.sum(axis=1) / 2
    return neutral


def output_paths(stock, output_dir=PARAMS_DIR):
    output_dir = Path(output_dir)
    return output_dir / f"{stock}_params.json", output_dir / f"{stock}_rl_params.json"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def calibrate(stock, output_dir=PARAMS_DIR, drift_offset=0.0):
    # Load the raw data
    messages, book = load_raw(stock, DEFAULT_DATE, RAW_DIR)
    date = DEFAULT_DATE

    # process the data and build the params dictionary 
    messages, book = process(messages, book, date)
    params = build_params(stock, date, messages, book, drift_offset)

    # Build the RL params 
    rl_params = build_rl_params(params)

    # Save the params to JSON files
    params_path, rl_params_path = output_paths(stock, output_dir)
    save_json(params_path, params)
    save_json(rl_params_path, rl_params)
    return params_path, rl_params_path, params, rl_params


def main():
    # Calibrate parameters for each stock and print some stats for sanity checking
    for stock in RUN_STOCKS:
        params_path, rl_params_path, params, _ = calibrate(stock, output_dir=RUN_OUTPUT_DIR, drift_offset=RUN_DRIFT_OFFSET)
        
        sigma = np.array(params["volatility"]["sigma_k"])
        spread = float(np.nanmean(params["book_shape"]["spread_by_bucket"]))
        
        print(f"{stock}: saved {params_path.name}, {rl_params_path.name}")
        print(f"  sigma: {np.nanmin(sigma):.2e} to {np.nanmax(sigma):.2e}")
        print(f" spread: ${spread:.4f}")


if __name__ == "__main__":
    main()
