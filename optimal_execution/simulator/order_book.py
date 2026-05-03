from collections import deque
from sortedcontainers import SortedDict

BID = 0
ASK = 1

class OrderBook:
    def __init__(self, tick_size=0.01):
        # Create a sorted dictionary for bids and asks
        # We store bids with negative prices to keep the descending order
        self.tick_size = tick_size
        self.bids = SortedDict()  
        self.asks = SortedDict()  
        self.orders = {}          
        self._next_id = 1

    def best_bid(self):
        # Find the best bid and return the true value
        if not self.bids:
            return 0.0
        return -self.bids.peekitem(0)[0]

    def best_ask(self):
        # Same for best ask
        if not self.asks:
            return float('inf')
        return self.asks.peekitem(0)[0]

    def mid_price(self):
        # Mid = (best_bid + best_ask) / 2
        if not self.bids or not self.asks:
            return 0.0
        return (self.best_bid() + self.best_ask()) / 2

    def spread(self):
        return self.best_ask() - self.best_bid()

    def add_limit_order(self, price, size, side, order_id=None):
        # Create an order id if not provided
        if order_id == None:
            order_id = self._next_id
            self._next_id += 1

        # Choose which side of the book to add to
        if side == BID:
            book, key = self.bids, -price
        else:
            book, key = self.asks, price

        # If the price level doesn't exist, create it
        if key not in book:
            book[key] = deque()

        # Add the order to the queue at this price level
        book[key].append((order_id, size))
        self.orders[order_id] = {'price': price, 'side': side, 'size': size}
        return order_id

    def cancel_order(self, order_id, size=None):
        # we cancel by order id 
        if order_id not in self.orders:
            return

        order = self.orders[order_id]

        # If size is None or greater than order size, remove the whole order
        if size is None or size >= order['size']:
            self._remove_order(order_id)
            return

        # Otherwise do a partial cancel, first we find the side and price level
        if order['side'] == BID:
            book, key = self.bids, -order['price']
        else:
            book, key = self.asks, order['price']

        # Walk through the queue until we find the order then update the size
        queue = book[key]
        for i, (oid, osize) in enumerate(queue):
            if oid == order_id:
                queue[i] = (order_id, osize - size)
                break

        self.orders[order_id]['size'] -= size

    def _remove_order(self, order_id):
        if order_id not in self.orders:
            return

        # Find price level like before
        order = self.orders[order_id]
        if order['side'] == BID:
            book, key = self.bids, -order['price']
        else:
            book, key = self.asks, order['price']

        # Walk through the queue until we find the order then remove it
        queue = book[key]
        for i, (oid, _) in enumerate(queue):
            if oid == order_id:
                del queue[i]
                break
        
        # if the queue is empty after removal, delete the price level
        if len(queue) == 0:
            del book[key]
        del self.orders[order_id]

    def execute_market_order(self, side, size):
        # for each side, get the orders and convert the prices to the true values
        if side == BID:
            book = self.asks
            get_price = lambda k: k
        else:
            book = self.bids
            get_price = lambda k: -k

        # Track how much of the MO is remaining and total cost
        remaining = size
        total_cost = 0.0

        # Walk through the book until we fill the MO or run out of orders
        while remaining > 0 and len(book) > 0:
            best_key = book.peekitem(0)[0]
            price = get_price(best_key)
            queue = book[best_key]

            while remaining > 0 and len(queue) > 0:
                order_id, order_size = queue[0]
                fill = min(remaining, order_size)

                total_cost += fill * price
                remaining -= fill

                # If the order is fully filled, remove it. Otherwise update the size
                if fill == order_size:
                    queue.popleft()
                    del self.orders[order_id]
                else:
                    queue[0] = (order_id, order_size - fill)
                    self.orders[order_id]['size'] -= fill

            # Remove the price level if it's empty after processing
            if len(queue) == 0:
                del book[best_key]

        filled = size - remaining
        if filled > 0:
            avg_price = total_cost / filled
        else: 
            avg_price = 0.0

        return {'filled': filled, 'unfilled': remaining, 'avg_price': avg_price, 'total_cost': total_cost}



    def get_state(self, n_levels=5):
        # Builds a snapshot of the top n levels of the book
        bid_prices, bid_sizes = [], []
        ask_prices, ask_sizes = [], []

        # Work through the bid levels from best to worst price, converting to true prices
        # and summing the order sizes at each level
        for i, (neg_price, queue) in enumerate(self.bids.items()):
            if i >= n_levels:
                break
            bid_prices.append(-neg_price)
            bid_sizes.append(sum(s for _, s in queue))

        for i, (price, queue) in enumerate(self.asks.items()):
            if i >= n_levels:
                break
            ask_prices.append(price)
            ask_sizes.append(sum(s for _, s in queue))

        # Fill any missing levels with zeros
        while len(bid_prices) < n_levels:
            bid_prices.append(0.0)
            bid_sizes.append(0)
        while len(ask_prices) < n_levels:
            ask_prices.append(0.0)
            ask_sizes.append(0)

        return {'bid_prices': bid_prices, 'bid_sizes': bid_sizes, 'ask_prices': ask_prices, 'ask_sizes': ask_sizes}