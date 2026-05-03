from pathlib import Path

import numpy as np
import pandas as pd

# File naming conventions
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "datasets" / "raw"
DEFAULT_DATE = "2012-06-21"
DEFAULT_STOCKS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
LOBSTER_START = "34200000"
LOBSTER_END = "57600000"

# LOBSTER data format
MESSAGE_COLUMNS = ["timestamp", "type", "order_id", "size", "price", "direction"]
TICK_SIZE = 0.01
FILTER_START = 36000
FILTER_END = 55800
BUCKET_SECONDS = 300
N_BUCKETS = 66
N_LEVELS = 20
BOOK_LEVELS = 5
MAX_REL_TICKS = 50

# Limit order side
SIDE = {"bid": 0, "ask": 1}
# Market order side
MARKET_ORDER_SIDE = {"bid": 1, "ask": 0}

ROUND_LOTS = [100, 200, 500, 1000]


def book_columns():
    # Create column headersfor the LOBSTER data
    columns = []

    for level in range(1, BOOK_LEVELS + 1):
        columns += [f"ask_price_{level}", f"ask_size_{level}", f"bid_price_{level}", f"bid_size_{level}"]

    return columns


def load_raw(stock, date=DEFAULT_DATE, raw_dir=RAW_DIR):
    # Load raw LOBSTER data for a given stock and date
    folder = Path(raw_dir) / f"LOBSTER_SampleFile_{stock}_{date}_{BOOK_LEVELS}"
    file_name = f"{stock}_{date}_{LOBSTER_START}_{LOBSTER_END}"
    msg_path = folder / f"{file_name}_message_{BOOK_LEVELS}.csv"
    book_path = folder / f"{file_name}_orderbook_{BOOK_LEVELS}.csv"

    # Read the message and order book data
    messages = pd.read_csv(msg_path, header=None, names=MESSAGE_COLUMNS)
    book = pd.read_csv(book_path, header=None, names=book_columns())

    # Convert all price columns in the book to dollars
    messages["price"] *= 1e-4
    price_columns = []
    for column in book.columns:
        if "price" in column:
            price_columns.append(column)
    book[price_columns] *= 1e-4

    return messages, book


def process(messages, book, date):
    # Compute mid price, spread and relative price level for each message,
    # then filter to the desired time range and relative price levels
    book["mid_price"] = (book["ask_price_1"] + book["bid_price_1"]) / 2
    book["spread"] = book["ask_price_1"] - book["bid_price_1"]
    messages["side"] = messages["direction"].map({1: "bid", -1: "ask"})
    relative_ticks = (messages["price"] - book["mid_price"]).abs() / TICK_SIZE
    messages["rel_level"] = relative_ticks.round().astype(int)

    time_mask = (messages["timestamp"] >= FILTER_START) & (messages["timestamp"] <= FILTER_END)
    level_mask = messages["rel_level"] <= MAX_REL_TICKS
    size_mask = messages["size"] > 0
    keep = time_mask & level_mask & size_mask
    
    # Filter both messages and book to the same time range, and update row indices
    messages = messages[keep].reset_index(drop=True)
    book = book[keep].reset_index(drop=True)

    # Assign each message and book entry to a 5-minute bucket
    bucket = ((messages["timestamp"] - FILTER_START) / BUCKET_SECONDS).clip(0, N_BUCKETS - 1)
    bucket = bucket.astype(int)
    messages["bucket"] = bucket
    book["bucket"] = bucket
    messages["date"] = date
    book["date"] = date

    return messages, book


def level_rates(messages, event_types):
    # Count events in each 5-minute bucket, price level, and side, as outlined in Chapter 5
    counts = np.zeros((N_BUCKETS, N_LEVELS, 2))

    # Filter by type (new limit order, partial cancel, full cancel, visble market order, hidden market order)
    # Then filter to only relevant price levels
    type_mask = messages["type"].isin(event_types)
    level_mask = messages["rel_level"] < N_LEVELS
    keep = type_mask & level_mask

    events = messages[keep].copy()
    events["side_num"] = events["side"].map(SIDE)

    # Walk through each filtered event and add one count to its
    # bucket / relative level / side slot.
    for bucket, level, side in zip(events["bucket"], events["rel_level"], events["side_num"]):
        bucket = int(bucket)
        level = int(level)
        side = int(side)
        counts[bucket, level, side] += 1

    # Convert counts per bucket into rates
    rates = counts / BUCKET_SECONDS
    return rates

# Then do the same thing for market orders
def market_order_rates(messages):
    counts = np.zeros((N_BUCKETS, 2))

    execution_mask = messages["type"].isin([4, 5])
    executions = messages[execution_mask].copy()
    executions["market_order_side"] = executions["side"].map(MARKET_ORDER_SIDE)

    # Count market orders by bucket and side.
    for bucket, side in zip(executions["bucket"], executions["market_order_side"]):
        bucket = int(bucket)
        side = int(side)
        counts[bucket, side] += 1

    # Convert counts per bucket into rates
    rates = counts / BUCKET_SECONDS
    return rates


def estimate_rates(messages):
    # Return a dictionary of the rates
    return {"lambda_limit": level_rates(messages, [1]),
            "lambda_cancel": level_rates(messages, [2, 3]),
            "lambda_mo": market_order_rates(messages)}


def estimate_volatility(book):
    # Calculate the realised volatility by bucket from mid-price log returns
    sigma = np.zeros(N_BUCKETS)

    # Compute vol for each bucket separately
    for bucket in range(N_BUCKETS):
        prices = book[book["bucket"] == bucket]["mid_price"].to_numpy()
        returns = np.diff(np.log(prices))
        sigma[bucket] = np.sqrt(np.sum(returns ** 2) / (len(returns) - 1))
        
    return sigma


def estimate_book_shape(book, levels=BOOK_LEVELS):
    depth = np.zeros((levels, 2))

    # Calculate the average depth at each level
    for level in range(levels):
        depth[level, 0] = np.mean(book[f"bid_size_{level + 1}"])
        depth[level, 1] = np.mean(book[f"ask_size_{level + 1}"])

    # Measure the bid ask spread across the day and compute a bucket average
    spread_by_bucket = np.zeros(N_BUCKETS)
    for bucket in range(N_BUCKETS):
        bucket_spread = book[book["bucket"] == bucket]["spread"]
        spread_by_bucket[bucket] = np.mean(bucket_spread)

    return {"depth": depth, "spread_by_bucket": spread_by_bucket}


def lognormal_fit(sizes):
    # Fit a simple log-normal model to execution sizes.
    log_sizes = np.log(sizes)
    mu = float(np.mean(log_sizes))
    sigma = float(np.std(log_sizes, ddof=1))
    n_samples = int(len(sizes))

    return {"mu": mu, "sigma": sigma, "n_samples": n_samples}


def hybrid_fit(sizes):
    # Exclude the round order sizes (lots) and fit a lognormal dist to the remaining
    # Check if the size is a round lot
    is_round = np.isin(sizes, ROUND_LOTS)
    # keep only round lots
    round_sizes = sizes[is_round]
    # keep only odd lots
    odd_sizes = sizes[~is_round]

    p_round = float(len(round_sizes) / len(sizes))

    odd_log_sizes = np.log(odd_sizes)

    # For each round lot, compute the proportion of all round lots that it represents
    round_lot_weights = {}
    for lot in ROUND_LOTS:
        lot_count = np.sum(round_sizes == lot)
        round_lot_weights[lot] = float(lot_count / len(round_sizes))

    # Fit a lognormal distribution to the odd lots
    odd_lot_mu = float(np.mean(odd_log_sizes))
    odd_lot_sigma = float(np.std(odd_log_sizes, ddof=1))

    return {"p_round": p_round,
            "round_lot_weights": round_lot_weights,
            "odd_lot_mu": odd_lot_mu,
            "odd_lot_sigma": odd_lot_sigma,
            "n_samples": int(len(sizes)),
            "n_round": int(len(round_sizes)),
            "n_odd": int(len(odd_sizes))}


def estimate_size_distribution(messages):
    # Filter only market orders, then fit a lognormal and hybrid distribution
    execution_rows = messages["type"].isin([4, 5])
    execution_sizes = messages[execution_rows]["size"].to_numpy()
    execution_sizes = execution_sizes[execution_sizes > 0]

    fit = lognormal_fit(execution_sizes)
    hybrid = hybrid_fit(execution_sizes)

    return {"mu": fit["mu"],
            "sigma": fit["sigma"],
            "n_samples": fit["n_samples"],
            "validation": {},
            "hybrid": hybrid}