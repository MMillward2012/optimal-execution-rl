import numpy as np
from simulator.order_book import BID, ASK


def generate_events(book, rates, dt, drift_ticks, rng, spread_ticks=0.0):
    """
    Generate one timestep of background market activity.
    """

    mid = book.mid_price()
    if mid <= 0:
        return

    tick = book.tick_size
    n_levels = rates['lambda_limit'].shape[0]

    # For each price level, place new limit orders (of size sampled from our hybrid distribution)
    # according to a Poisson proces with the given rates.
    for level in range(n_levels):
        for side in [BID, ASK]:
            rate = rates['lambda_limit'][level, side]

            for k in range(rng.poisson(max(0, rate * dt))):
                # Sample the price for the new order
                price = _limit_order_price(mid, tick, side, level, drift_ticks, spread_ticks)
                # Sample the size for the new order
                size = _sample_size(rng, rates.get('size_distribution'))

                # Add the order
                if price > 0:
                    book.add_limit_order(price, size, side)

    # Randomly cancel orders from the book using the given Poisson rates for each level
    for level in range(n_levels):
        for side in [BID, ASK]:
            rate = rates['lambda_cancel'][level, side]

            for k in range(rng.poisson(max(0, rate * dt))):
                _cancel_at_level(book, side, level)

    mo_rates, mo_size_distribution = _market_order_inputs(rates)

    # Parent market-order calibration matches the simulator's book-walking MO.
    for side in [BID, ASK]:
        rate = mo_rates[side]

        for k in range(rng.poisson(max(0, rate * dt))):
            size = _sample_size(rng, mo_size_distribution)
            book.execute_market_order(side, size)


def _market_order_inputs(rates):
    """Prefer parent-market-order calibration, with old params as fallback."""
    parent_rates = rates.get('lambda_parent_mo')
    if parent_rates is not None:
        parent_sizes = rates.get('parent_mo_size_distribution')
        return parent_rates, parent_sizes or rates.get('size_distribution')

    return rates['lambda_mo'], rates.get('size_distribution')


def _limit_order_price(mid, tick, side, level, drift_ticks, spread_ticks):
    """
    Choose the price for a new limit order.
    
    To account for drift, we shift the incoming prices by drift_ticks
    """
    base_level = level + 1

    if side == BID:
        # Positive drift makes bids more aggressive.
        effective_level = base_level + spread_ticks - drift_ticks
        effective_level = max(1.0, effective_level)
        price = mid - effective_level * tick
    else:
        # Positive drift makes asks less aggressive.
        effective_level = base_level + spread_ticks + drift_ticks
        effective_level = max(1.0, effective_level)
        price = mid + effective_level * tick

    return round(price / tick) * tick # Round to nearest tick (note the tick scale changes so we dont just use round(price))


def _sample_size(rng, size_distribution=None):
    """Sample the size for a new order from our hybrid distribution."""
    if size_distribution == None:
        if rng.rand() < 0.3:
            return 100
        return max(1, int(rng.lognormal(4.5, 0.8)))

    if size_distribution.get('model') == 'empirical':
        values = [int(value) for value in size_distribution['values']]
        weights = list(size_distribution['weights'])
        return int(rng.choice(values, p=weights))

    if rng.rand() < size_distribution['p_round']:
        lots = [int(lot) for lot in size_distribution['round_lot_weights'].keys()]
        weights = list(size_distribution['round_lot_weights'].values())
        return int(rng.choice(lots, p=weights))

    odd_lot_mu = size_distribution['odd_lot_mu']
    odd_lot_sigma = size_distribution['odd_lot_sigma']
    odd_lot_size = np.exp(rng.normal(odd_lot_mu, odd_lot_sigma))
    return max(1, int(round(odd_lot_size)))


def _cancel_at_level(book, side, level):
    """Cancel one order from the current visible book level."""
    if side == BID:
        levels = book.bids
    else:
        levels = book.asks

    if level >= len(levels):
        return

    _, queue = levels.peekitem(level)
    if len(queue) == 0:
        return

    order_id = queue[0][0]
    book.cancel_order(order_id)
