from market.agent import Agent
from copy import deepcopy
import heapq as hq
from orders import OrderBook
from util import ft


class ExchangeAgent(Agent):
    """ Represents the exchange.  Can handle multiple symbols each with its own order book.
        Understands placing market and limit orders, and reducing/canceling limit orders.
        Supports order expiration times (good until).  Maintains and delivers LOB snapshots
        and a configurable LOB snapshot history. """

    def __init__(self, symbols, args):

        super().__init__(0)
        self.args, self.symbols = args, symbols
        self.reset()

    def reset(self):
        self.book = { s: OrderBook(s, self.args) for s in self.symbols }
        self.notify = []

    def message(self, ct, msg):
        self.ct = ct
        mtype = msg['type']

        if mtype in ['exopen', 'exclose']: return

        # Handle subscription requests.
        if mtype == 'subscribe':
            hq.heappush(self.notify, (ct, msg['aid'], msg['levels'], msg['intvl']))
            return

        # All messages except exopen, exclose, subscribe contain an order.
        order = msg['order']

        # History is executed pre-market.  Nothing executes after hours.
        if (ct < self.args.exch_open and order.aid > 0) or ct > self.args.exch_close: return

        # Extract the relevant book for the current order.  Update its current time.
        book = self.book[order.symbol]
        book.ct = ct

        # Update fundamental if history order.
        if order.aid < 0 and 'fund' in msg: book.fund = msg['fund']

        # Handle the order
        ### TEMP HACK: allow 'fund' mytype that only updates fundamental and nothing else.
        if mtype == 'fund':
            pass

        elif mtype == 'place':
            matching = True

            while matching:
                match = book.execute(order)
                if match:      # Order was at least partially matched.  Report out and continue.
                    filled = deepcopy(order)
                    filled.quantity = -match.quantity
                    filled.fill = match.fill
                    order.quantity -= filled.quantity

                    # Don't send execute messages to the history "agent".
                    if order.aid > 0: yield order.aid, { 'type': 'executed', 'order': filled }
                    if match.aid > 0: yield match.aid, { 'type': 'executed', 'order': match }

                    # Stop if order was completely exhausted.
                    if order.quantity == 0: matching = False
                else:
                    if order.limit is None:    # Market order couldn't be filled.
                        print (f"Warning: at {ft(ct)}, failed to fill market order {order}")
                        if order.aid > 0: yield order.aid, { 'type': 'rejected', 'order': order }
                    else:
                        book.enter(deepcopy(order))    # Add unfilled limit order to LOB.
                        if order.aid > 0: yield order.aid, { 'type': 'accepted', 'order': order }
                    matching = False

        elif mtype == 'cancel':
            cancelled = book.cancel(order)    # Cancellation could fail if executed already.
            result = 'cancelled' if cancelled else 'cancel_failed'
            if order.aid > 0: yield order.aid, { 'type': result, 'order': order }

        elif mtype == 'reduce':
            reduced = book.reduce(order)      # Reduction could fail if executed already.
            result = 'reduced' if reduced else 'reduce_failed'
            if order.aid > 0: yield order.aid, { 'type': result, 'order': order }

        # If market not open or no subscriptions, done.  Else send LOB subscriptions as needed.
        if ct < self.args.exch_open or len(self.notify) == 0: return

        # Build a snapshot to send to all agents in the subscription queue whose timer has
        # expired.  Each agent can request a different number of levels.
        if ct > self.notify[0][0]: book.clearexp()    # Clear expired orders before notifying.
        while ct > self.notify[0][0]:
            t, aid, lev, intvl = hq.heappop(self.notify)
            snap = book.snapview(lev)
            hq.heappush(self.notify, (ct + intvl, aid, lev, intvl))

            # Only send if we weren't processing this agent's own order.
            if aid != order.aid:
                yield aid, { 'type': 'lob', 'snap': snap, 'fund': book.fund, 'hist': book.snap_hist }

        # Uncomment this line to display an LOB snapshot after each inbound order is processed.
        #print(ft(ct), "\n" + book.summary(10))

