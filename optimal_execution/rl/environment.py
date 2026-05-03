import json
from pathlib import Path
import numpy as np

from simulator.order_book import OrderBook, BID, ASK
from simulator.background_order_flow import generate_events

# This file is pretty long as it contains the full environment implementation
# None of the details are 

def load_params(ticker="AAPL"):

    params_dir = Path(__file__).parent.parent / "datasets" / "params"
    path = params_dir / f"{ticker}_rl_params.json"

    with open(path) as f:
        return json.load(f)


def make_env_from_params(ticker="AAPL", total_shares=12000, max_steps=50,
                              steps_per_action=20, risk_aversion=0.0025,
                              urgency_coeff=0.0, urgency_shape="linear",
                              impact_coeff=0.01, warmup_steps=500, seed=None):

    params = load_params(ticker)
    return ExecutionEnv(
        params=params,
        total_shares=total_shares,
        max_steps=max_steps,
        steps_per_action=steps_per_action,
        risk_aversion=risk_aversion,
        urgency_coeff=urgency_coeff,
        urgency_shape=urgency_shape,
        impact_coeff=impact_coeff,
        warmup_steps=warmup_steps,
        seed=seed)


def _clamp_shares(shares, inventory):
    """Return a valid non-negative share count bounded by inventory."""
    return max(0, min(int(shares), int(inventory)))


def _spread_to_extra_ticks(spread, tick_size):
    extra_ticks = spread / tick_size / 2 - 1
    return max(0.0, extra_ticks)


def _regime_setup_from_params(params):
    """
    Pull the regime setup out of the params file.

    RL params should already contain open / midday / close. If someone gives a
    raw calibrated params file, this builds a bucket-by-bucket fallback.
    """
    tick_size = params["metadata"]["tick_size"]
    n_buckets = params["metadata"]["n_buckets"]
    regime_by_bucket = params.get("regime_by_bucket")
    rates_by_regime = {}
    spread_ticks_by_regime = {}
    sigma_by_regime = {}
    size_distribution = params.get("size_distribution", {}).get("hybrid")
    parent_market_order_distribution = params.get("size_distribution", {}).get("parent_market_order", {})
    parent_size_distribution = (
        parent_market_order_distribution.get("empirical")
        or parent_market_order_distribution.get("hybrid")
    )

    if regime_by_bucket is not None:
        for name, regime in params["regimes"].items():
            rates_by_regime[name] = {
                "lambda_limit": np.array(regime["lambda_limit"]),
                "lambda_cancel": np.array(regime["lambda_cancel"]),
                "lambda_mo": np.array(regime["lambda_mo"]),
                "size_distribution": size_distribution,
            }
            if "lambda_parent_mo" in regime:
                rates_by_regime[name]["lambda_parent_mo"] = np.array(regime["lambda_parent_mo"])
                rates_by_regime[name]["parent_mo_size_distribution"] = parent_size_distribution

            if "spread_ticks" in regime:
                spread_ticks_by_regime[name] = regime["spread_ticks"]
            else:
                spread_ticks_by_regime[name] = _spread_to_extra_ticks(regime["spread"], tick_size)

            sigma_by_regime[name] = regime["sigma"]

        return regime_by_bucket, rates_by_regime, spread_ticks_by_regime, sigma_by_regime

    limit_rates = np.array(params["arrival_rates"]["lambda_limit"])
    cancel_rates = np.array(params["arrival_rates"]["lambda_cancel"])
    mo_rates = np.array(params["arrival_rates"]["lambda_mo"])
    parent_mo_raw = params["arrival_rates"].get("lambda_parent_mo")
    parent_mo_rates = np.array(parent_mo_raw) if parent_mo_raw is not None else None
    spread_by_bucket = np.array(params["book_shape"]["spread_by_bucket"])
    sigma_by_bucket = np.array(params["volatility"]["sigma_k"])
    regime_by_bucket = []

    for bucket in range(n_buckets):
        name = "bucket_" + str(bucket)
        regime_by_bucket.append(name)
        rates_by_regime[name] = {
            "lambda_limit": limit_rates[bucket],
            "lambda_cancel": cancel_rates[bucket],
            "lambda_mo": mo_rates[bucket],
            "size_distribution": size_distribution,
        }
        if parent_mo_rates is not None:
            rates_by_regime[name]["lambda_parent_mo"] = parent_mo_rates[bucket]
            rates_by_regime[name]["parent_mo_size_distribution"] = parent_size_distribution
        spread_ticks_by_regime[name] = _spread_to_extra_ticks(spread_by_bucket[bucket], tick_size)
        sigma_by_regime[name] = sigma_by_bucket[bucket]

    return regime_by_bucket, rates_by_regime, spread_ticks_by_regime, sigma_by_regime


class ExecutionEnv:

    N_ACTIONS = 51
    ACTION_FRACTIONS = np.linspace(0.0, 1.0, N_ACTIONS)
    IMPACT_NORM = 0.001
    DEPTH_CAP_MULTIPLIER = 3.0
    REPAIR_LEVELS = 5
    REPAIR_ORDER_SIZE = 100

    def __init__(
        self,
        params,
        total_shares=12000,
        max_steps=50,
        steps_per_action=20,
        risk_aversion=0.0025,
        urgency_coeff=0.0,
        urgency_shape="linear",
        impact_coeff=0.01,
        warmup_steps=500,
        seed=None,
    ):
        self.params = params
        self.seed = seed

        self.total_shares = total_shares
        self.max_steps = max_steps
        self.steps_per_action = steps_per_action
        self.risk_aversion = risk_aversion
        self.urgency_coeff = urgency_coeff
        self.urgency_shape = urgency_shape
        self.impact_coeff = impact_coeff
        self.warmup_steps = warmup_steps
        self.state_dim = 6   # Change this if adding vol, drift or alpha feature

        # Extract the calibrated market parameters we need
        meta = params['metadata']
        self.tick_size = meta['tick_size']
        self.opening_mid = meta['opening_mid']
        self.n_buckets = meta['n_buckets']
        self.calibrated_drift = meta.get('drift_offset', 0.0)
        spread_by_bucket = params.get('book_shape', {}).get('spread_by_bucket', [self.tick_size])
        self.avg_spread = float(meta.get('neutral_spread', np.mean(spread_by_bucket)))
        self.regime_by_bucket, self.rates_by_regime, self.spread_ticks_by_regime, self.sigma_by_regime = _regime_setup_from_params(params)

        # Average depth for synthetic tail
        self.depth_shape = np.array(params['book_shape']['depth'])
        if len(self.depth_shape) >= 3:
             self.avg_depth = float(np.sum(self.depth_shape[:3]))
        else:
            self.avg_depth = 300.0

        ##################################################### delete
        # Parameters to reset the environment
        self.book = None
        self.rng = None
        self.inventory = 0
        self.current_step = 0
        self.initial_mid = 0.0
        self.cumulative_impact = 0.0
        self.last_exec_impact = 0.0
        self.shares_sold = 0
        self.episode_cash = 0.0
        self.episode_risk_penalty = 0.0

    def reset(self, seed=None):
        if seed != None:
            self.seed = seed

        self.rng = np.random.RandomState(self.seed)

        # Create a new order book
        self.book = OrderBook(tick_size=self.tick_size)
        self._seed_book()
        self._warm_up_book()

        # Episode state
        self.inventory = self.total_shares
        self.current_step = 0
        self.initial_mid = self.book.mid_price()
        self.cumulative_impact = 0.0
        self.last_exec_impact = 0.0
        self.shares_sold = 0
        self.episode_cash = 0.0
        self.episode_risk_penalty = 0.0

        return self._get_state()

    def _warm_up_book(self):
        """Create some background order flow before the agent starts trading."""
        if self.warmup_steps <= 0:
            return

        # Warm up using the open regime settings so the book continues from a realistic state
        regime = self.regime_by_bucket[0]
        rates = self.rates_by_regime[regime]
        spread_ticks = self.spread_ticks_by_regime[regime]

        for k in range(self.warmup_steps):
            # Take a reference before the book is evolved in case it gets depleted
            reference = self._book_reference()
            generate_events(
                self.book,
                rates,
                dt=1.0,
                drift_ticks=0.0,
                rng=self.rng,
                spread_ticks=spread_ticks,
            )
            self._repair_depleted_book(reference)
            self._control_book_depth()

    def step(self, action, exact_shares=None):
        """
        When the agent picks an action, this function executes it and evolves the state
        """

        # Get the old mid price and previous inventory
        mid_before = self.book.mid_price()
        inventory_before = self.inventory

        # Check which time bucket the agent is in and use those parameters
        bucket = min(self.current_step * self.n_buckets // self.max_steps, self.n_buckets - 1)
        regime = self.regime_by_bucket[bucket]
        rates = self.rates_by_regime[regime]
        spread_ticks = self.spread_ticks_by_regime[regime]


        # Compute the risk penalty term with urgency.
        # Typically use cubic urgency because it interferes less with the agents trading schedule
        # and enforces the agent gets punished accordingly if it hasnt sold all shares by the end
        sigma_return = self.sigma_by_regime[regime]
        sigma = sigma_return * self.initial_mid
        progress = self.current_step / self.max_steps
        if self.urgency_shape == "quadratic":
            risk_multiplier = 1.0 + self.urgency_coeff * (progress ** 2)
        elif self.urgency_shape == "cubic":
            risk_multiplier = 1.0 + self.urgency_coeff * (progress ** 3)
        else:
            risk_multiplier = 1.0 + self.urgency_coeff * progress
        base_risk_penalty = self.risk_aversion * (sigma ** 2) * (inventory_before ** 2)
        risk_penalty = base_risk_penalty * risk_multiplier

        # Compute the number of shares the agent wants to sell and ensure its valid
        if exact_shares == None:
            action_fraction = float(self.ACTION_FRACTIONS[action])
            shares = _clamp_shares(self.inventory * action_fraction, self.inventory)
        else:
            shares = _clamp_shares(exact_shares, self.inventory)
            action_fraction = shares / self.inventory if self.inventory > 0 else 0.0

        # Execute agent's sell order
        exec_result = self._execute_sell(shares)
        self.inventory -= exec_result['filled']
        self.episode_cash += exec_result['filled'] * exec_result['avg_price']

        # Update the drift
        drift = self.calibrated_drift + self.cumulative_impact

        # Run the simulator for a few steps to evolve the book
        for k in range(self.steps_per_action):
            # Same as before, take a reference in case the book is depleted
            reference = self._book_reference()
            generate_events(
                self.book,
                rates,
                dt=1.0,
                drift_ticks=drift,
                rng=self.rng,
                spread_ticks=spread_ticks,
            )
            self._repair_depleted_book(reference)
            self._control_book_depth()

        self.current_step += 1

        # Reward = negative execution cost - risk penalty (including urgency)
        execution_cost = exec_result['filled'] * (mid_before - exec_result['avg_price'])
        reward = -execution_cost - risk_penalty
        self.episode_risk_penalty += risk_penalty

        if self.current_step >= self.max_steps or self.inventory <= 0:
            done = True
        else:
            done = False

        terminal_exec = {'filled': 0, 'avg_price': 0.0}
        terminal_execution_cost = 0.0

        # Force liquidate at end and update the reward to include it
        if done and self.inventory > 0:
            terminal_mid = self.book.mid_price()

            if terminal_mid <= 0:
                terminal_mid = self._book_reference()['mid']

            final_exec = self._execute_sell(self.inventory)
            terminal_exec = final_exec

            if final_exec['filled'] > 0:
                terminal_execution_cost = final_exec['filled'] * (terminal_mid - final_exec['avg_price'])
                reward -= terminal_execution_cost
                self.episode_cash += final_exec['filled'] * final_exec['avg_price']
                self.inventory -= final_exec['filled']

        info = {'step': self.current_step,
                'action_fraction': action_fraction,
                'requested_shares': shares,
                'filled': exec_result['filled'],
                'avg_price': exec_result['avg_price'],
                'shares_sold': exec_result['filled'] + terminal_exec['filled'],
                'execution_cost': execution_cost,
                'terminal_filled': terminal_exec['filled'],
                'terminal_avg_price': terminal_exec['avg_price'],
                'terminal_execution_cost': terminal_execution_cost,
                'mid': self.book.mid_price(),
                'mid_price': self.book.mid_price(),
                'inventory_before': inventory_before,
                'inventory': self.inventory,
                'risk_penalty': self.episode_risk_penalty,
                'step_risk_penalty': risk_penalty,
                'risk_multiplier': risk_multiplier,
                'regime': regime,
                'drift_used': drift}

        # Also log the IS if the episode is done
        if done:
            info['implementation_shortfall'] = (self.total_shares * self.initial_mid - self.episode_cash)

        # Move to the next state, give the reward and indicate if the episode is done
        return self._get_state(), reward, done, info



    def _execute_sell(self, shares):
        """Execute agent's sell order with synthetic tail if the order is not fully filled"""
        if shares <= 0:
            self.last_exec_impact = 0.0
            return {'filled': 0, 'avg_price': 0.0}

        reference = self._book_reference()
        result = self.book.execute_market_order(ASK, shares)

        # If the order is not fully filled, we provide a worse price for any remaining shares
        if result['unfilled'] > 0:
            # Get the price using synthetic tail
            tail_price = self._synthetic_tail_price(result['unfilled'], reference)
            total_filled = result['filled'] + result['unfilled']
            total_value = result['filled'] * result['avg_price'] + result['unfilled'] * tail_price
            result = {'filled': total_filled, 'avg_price': total_value / total_filled}

        self._repair_depleted_book(reference)

        # Apply permanent impact
        if result['filled'] > 0:
            if reference['mid'] > 0:
                self.last_exec_impact = (result['avg_price'] - reference['mid']) / reference['mid']
            else:
                self.last_exec_impact = 0.0
            
            # Use permanent impact formula from (i think) Section 5.1.4
            old_impact = self.impact_coeff * np.sqrt(self.shares_sold / 10000)
            new_impact = self.impact_coeff * np.sqrt((self.shares_sold + result['filled']) / 10000)
            impact = new_impact - old_impact
            self.cumulative_impact -= impact 
            self.shares_sold += result['filled']
        else:
            self.last_exec_impact = 0.0

        return result


    def _synthetic_tail_price(self, size, reference):
        """Price for unfilled order, deteriorating with size."""

        best_bid = reference['bid']
        if best_bid <= 0:
            best_bid = reference['mid'] - self.tick_size

        # Linear deterioration based on size relative to typical depth
        ticks_down = (size / self.avg_depth) * 10  # 10 ticks per avg_depth
        return max(self.tick_size, best_bid - ticks_down * self.tick_size)

    def _book_reference(self):
        """Take a pre-trade reference before the book can be depleted."""
        mid = self.book.mid_price()
        if mid <= 0:
            mid = self.opening_mid

        return {'mid': mid, 'bid': self.book.best_bid(), 'ask': self.book.best_ask()}

    def _repair_depleted_book(self, reference):
        """
        Add a small amount of worse-priced liquidity if one side is empty.

        This shouldnt need to be called but is a fail safe in case the book gets completely depleted.
        Not 100% realistic to true markets but it prevents the agent from exploiting the simulator book dynamics
        """
        # If the bid side is empty, add some orders at worse prices below the old best bid
        # We add REPAIR_ORDER_SIZE number of limit orders at prices below the reference bid
        if len(self.book.bids) == 0:
            start_bid = reference['bid']
            if start_bid <= 0:
                start_bid = reference['mid'] - self.tick_size

            for level in range(1, self.REPAIR_LEVELS + 1):
                price = start_bid - level * self.tick_size
                if price > 0:
                    self.book.add_limit_order(price, self.REPAIR_ORDER_SIZE, BID)

        # Do the converse for the ask side
        if len(self.book.asks) == 0:
            start_ask = reference['ask']
            if not np.isfinite(start_ask):
                start_ask = reference['mid'] + self.tick_size

            for level in range(1, self.REPAIR_LEVELS + 1):
                price = start_ask + level * self.tick_size
                if price > 0:
                    self.book.add_limit_order(price, self.REPAIR_ORDER_SIZE, ASK)


    def _control_book_depth(self):
        """
        This just removes any old accumulated volume that may have not been removed during warmup
        """
        self._trim_side_depth(self.book.bids, BID)
        self._trim_side_depth(self.book.asks, ASK)

    def _trim_side_depth(self, levels, side):
        if len(self.depth_shape) == 0:
            return

        # Loop through each price level and trim the total size at that level to be within 
        # a reasonable multiple of the calibrated depth at that level
        last_level = len(self.depth_shape) - 1
        for level, key in enumerate(list(levels.keys())):
            # Look up how much depth we typically have at this level
            calibrated_depth = float(self.depth_shape[min(level, last_level), side])
            # Set a maximum value allowed
            cap = max(self.REPAIR_ORDER_SIZE, int(self.DEPTH_CAP_MULTIPLIER * calibrated_depth))

            queue = levels.get(key)
            if queue is None:
                continue
            
            # get the total volume at this level and remove prices starting at the worst 
            # price until we are under the cap
            total = sum(size for k, size in queue)
            while total > cap and len(queue) > 0:
                order_id, size = queue[-1]
                excess = total - cap

                if size <= excess:
                    queue.pop()
                    total -= size
                    if order_id in self.book.orders:
                        del self.book.orders[order_id]
                    continue

                new_size = size - excess
                queue[-1] = (order_id, new_size)
                if order_id in self.book.orders:
                    self.book.orders[order_id]['size'] = new_size
                total = cap

            # If the level has no orders left, remove it from the book
            if len(queue) == 0 and key in levels:
                del levels[key]

    def _seed_book(self):
        """Initialise limit order book at the start of each episode"""
        mid = self.opening_mid
        first_regime = self.regime_by_bucket[0]
        spread_ticks = self.spread_ticks_by_regime[first_regime]

        # Build up to 20 price levels on each side
        for level in range(1, min(20, len(self.depth_shape)) + 1):
            for side in [BID, ASK]:
                level_from_mid = level + spread_ticks
                if side == BID:
                    price = mid - level_from_mid * self.tick_size
                else:
                    price = mid + level_from_mid * self.tick_size
                
                # keep adding orders until the total depth matches the calibrated depth
                target = int(self.depth_shape[level - 1, side])
                placed = 0
                while placed < target:
                    # Add orders in 100 share chunks initially
                    # Regular order flow is sampled using our calibrated data but for 
                    # initialisation this is quick and works
                    size = min(100, target - placed)
                    self.book.add_limit_order(price, size, side)
                    placed += size

    def _get_state(self):
        """Return state for agent. Here we compute all features"""

        mid = self.book.mid_price()
        if mid > 0:
            spread = self.book.spread()
        else:
            spread = 0.0

        book_state = self.book.get_state(n_levels=3)
        bid_depth = float(np.sum(book_state['bid_sizes']))
        ask_depth = float(np.sum(book_state['ask_sizes']))
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            imbalance = (bid_depth - ask_depth) / total_depth
        else:
            imbalance = 0.0

        if self.avg_depth > 0:
            relative_depth_raw = total_depth / self.avg_depth
        else:
            relative_depth_raw = 0.0

        relative_depth = np.clip(np.log1p(max(0.0, relative_depth_raw)), 0.0, 5.0)

        if self.avg_spread > 0:
            spread_norm = min(spread / self.avg_spread, 10.0)
        else:
            spread_norm = 0.0

        impact = np.clip(self.last_exec_impact / self.IMPACT_NORM, -5.0, 5.0)

        return np.array([
            self.inventory / self.total_shares,  
            1.0 - self.current_step / self.max_steps, 
            spread_norm,
            imbalance,
            relative_depth,
            impact,
        ], dtype=np.float32) # Use float32 for PyTorch compatibility
