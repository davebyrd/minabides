from copy import deepcopy
from dataclasses import dataclass

@dataclass
class OrderLevel:
    """ Holds an aggregated order level for an order book snapshot. """
    price: int          # whole cents
    quantity: int       # shares

@dataclass
class Order:
    """ Holds all the information for a single order. """
    next_oid = 0        # shared class variable for autoincrements
    aid: int            # originating agent id
    quantity: int       # shares
    symbol: str         # stock symbol
    limit: int = None   # limit price in whole cents
    fill: int = None    # filled price in whole cents
    oid: str = None     # unique order id (place) or existing order id (reduce/cancel)
    tag: str = None     # optional freeform tag for agent use
    exp: int = None     # optional expiration time (ns from midnight)

    def __post_init__(self):
        """ After initializing order, store original quantity (because it
            is decremented by exchange during partial processing).  Also
            add a unique auto-incrementing order id if one wasn't provided. """
        self.orig_qty = self.quantity
        if self.oid is None:
            self.oid = f"a{Order.next_oid}"
            Order.next_oid += 1

class OrderBook:
    """ Holds both sides of the LOB for a single symbol.  The book is simultaneously
        tracked in aggregated form (for snapshot performance) and not (for order
        processing).  Trigger logic on all order activity makes the minimal delta
        adjustments to the snapshot in real time, so the snapshot view is always current. """

    def __init__(self, symbol, args):
        self.symbol, self.args = symbol, args
        self.ct = 0                             # last time an order affected this book
        self.last_snap_hist = 0                 # snapshot history last updated (ns since midnight)

        self.fund = None                        # current fundamental value of this symbol
        self.book = { 'bid': [], 'ask': [] }    # full order book view (levels of individual orders)
        self.snap = { 'bid': [], 'ask': [] }    # snapshot view (levels of aggregate price, quantity)
        self.snap_hist = []                     # history of recent snapshots at some interval

        self.book_to_log, self.last_trade = None, [None, None]


    def execute(self, order):
        """ Attempts to partially or fully match and execute an incoming order.
            Updates made to book if required.  Then snap is updated to stay in sync,
            if changes were made. """

        buy = order.quantity >= 0       # Positive quantity means buy, negative means sell.
    
        # Grab the appropriate side of the LOB to avoid duplicate logic later on.
        if buy: book, snap = self.book['ask'], self.snap['ask']
        else:   book, snap = self.book['bid'], self.snap['bid']
    
        # If this side of the book is entirely empty, we can't execute.  Entering a limit
        # order will be handled separately by the exchange after matching fails.
        if not book: return None

        # The incoming order can only match against the oldest order at the best price level.
        match = book[0][0]
        while (match.exp is not None and match.exp <= self.ct):
            # Experimental feature: silently drop expired orders.  No notifications.
            # Only checked/dropped when they would have matched for performance reasons.
            expired = book[0].pop(0)

            if book[0]: snap[0].quantity -= abs(expired.quantity)       # still orders at level
            else: del book[0], snap[0]                                  # no more orders at level

            if not book: return None            # book now empty
            match = book[0][0]                  # get next order to try to match
    
        # Execute incoming order vs match if market or has compatible limit price.
        if buy: # match best ask
            if order.limit is None or order.limit >= match.limit:
                consume = order.quantity + match.quantity >= 0
            else: return None
        else:   # match best bid
            if order.limit is None or order.limit <= match.limit:
                consume = order.quantity + match.quantity <= 0
            else: return None
    
        if consume: # matched order entirely consumed, remove it
            book[0].pop(0)
            if book[0]: snap[0].quantity -= abs(match.quantity)
            else: del book[0], snap[0]
        else:       # matched order partially consumed, update quantity
            match = deepcopy(match)
            match.quantity = -order.quantity
            book[0][0].quantity += order.quantity
            snap[0].quantity -= abs(order.quantity)
    
        # Set filled price as limit price of incumbent (matched) order.
        match.fill = match.limit
    
        # If we made it here, there was a transaction, so log it.
        self.last_trade = [match.fill, abs(match.quantity)]

        # Update snapshot history if it is time.
        self.update_snaphist()

        return match
    
    
    def enter(self, order):
        """ Enter a limit order into the appropriate side of the LOB. """
    
        buy = order.quantity >= 0
        if buy: book, snap = self.book['bid'], self.snap['bid']
        else: book, snap = self.book['ask'], self.snap['ask']
    
        # If book is empty, or limit price is beyond current "worst" price, append new level.
        if not book or (buy and order.limit < book[-1][0].limit) or (not buy and order.limit > book[-1][0].limit):
            snap.append(OrderLevel(order.limit, abs(order.quantity)))
            book.append([order])
        else:   # We must search within book to find or insert correct price level.
            for i, o in enumerate(book):
                if order.limit == o[0].limit:   # Found correct level, already existed.  Add order.
                    snap[i].quantity += abs(order.quantity)
                    book[i].append(order)
                    break
                elif (buy and order.limit > o[0].limit) or (not buy and order.limit < o[0].limit):
                    # Current level is past the limit price.  Stop and insert new level here.
                    snap.insert(i, OrderLevel(order.limit, abs(order.quantity)))
                    book.insert(i, [order])
                    break

        # Update snapshot history if it is time.
        self.update_snaphist()
    
    
    def cancel(self, order):
        """ Attempt to cancel an existing order from the LOB. """
        buy = order.quantity >= 0
        if buy: book, snap = self.book['bid'], self.snap['bid']
        else: book, snap = self.book['ask'], self.snap['ask']
    
        success = False

        # Search the LOB for the order.  This can be slow.
        for i, o in enumerate(book):
            if order.limit == o[0].limit:
                for ci, co in enumerate(book[i]):
                    if order.oid == co.oid:             # Found the order
                        cancelled = book[i].pop(ci)
                        success = True
    
                        # If other orders at same level, decrement snapshot quantity.
                        if book[i]: snap[i].quantity -= abs(cancelled.quantity)
                        else: del book[i], snap[i]  # No other order at level, delete.

        # Update snapshot history if it is time.
        self.update_snaphist()

        return success
    
    
    def reduce(self, order):
        """ Attempt to reduce quantity of an existing order in LOB. """
        buy = order.quantity >= 0
        if buy: book, snap = self.book['bid'], self.snap['bid']
        else: book, snap = self.book['ask'], self.snap['ask']
    
        success = False
    
        # Search for order.
        for i, o in enumerate(book):
            if order.limit == o[0].limit:
                for mi, mo in enumerate(book[i]):
                    if order.oid == mo.oid:
                        # If request reduces quantity to 0, let cancel handle it.
                        if abs(mo.quantity) <= abs(order.quantity): self.cancel(order)
                        else:
                            # Reduce order quantity as requested.
                            snap[i].quantity -= abs(order.quantity)
                            mo.quantity -= order.quantity
                        success = True
    
        # Update snapshot history if it is time.
        self.update_snaphist()

        return success
    

    def clearexp(self, levels=1):
        """ Searches a certain number of levels of the order book to clean up
            expired orders.  Significant performance impact, so use advisedly.
            Typically called with same number of levels as snapshots about to
            be sent, to ensure expired orders are not advertised to anyone. """

        for book, snap in zip([self.book['ask'],self.book['bid']],[self.snap['ask'],self.snap['bid']]):
            if not book: continue

            booklen = len(book[0])
            book[0] = [ x for x in book[0] if x.exp is None or x.exp > self.ct ]
            dropped = booklen - len(book[0])
            #if dropped > 0: print (f"Dropped {dropped} expired orders.")
            if book[0]: snap[0].quantity = sum([ abs(x.quantity) for x in book[0] ])
            else: del book[0], snap[0]
    

    def snapview(self, levels=10):
        """ Quickly extract a certain number of levels from the LOB snapshot, which
            is always kept up to date by internal trigger logic. """

        return { 'bid': self.snap['bid'][:levels], 'ask': self.snap['ask'][:levels] }

    def update_snaphist(self):
        """ If LOB interval has passed, appends a current snapshot to the LOB history,
            then truncates it if necessary.  Also drops the current snapshot into
            the book log file. """

        if self.ct - self.last_snap_hist > self.args.lobintvl:
            self.clearexp()     # Expired orders should not make it into the history.
            self.snap_hist.append([x.quantity for x in self.snap['bid'][:self.args.levels]] +
                                  [x.quantity for x in self.snap['ask'][:self.args.levels]])
            self.snap_hist = self.snap_hist[-self.args.seqlen:]
            self.last_snap_hist = self.ct

            self.book_to_log = self.snaplog()
    
    def summary(self, levels=10):
        """ Build a pretty-printed string showing requested levels of the LOB. """
        s = f"{'BID':>6} {'PRICE':>8} {'ASK':>6}\n{'-'*3:>6} {'-'*5:>8} {'-'*3:>6}\n"
        for a in reversed(self.snap['ask'][:levels]): s += f"{'':6} {a.price/100:8.2f} {a.quantity:6d}\n"
        for b in self.snap['bid'][:levels]: s += f"{b.quantity:6d} {b.price/100:8.2f}\n"
        return s

    def snaplog(self):
        """ Return the time plus the configured number of order book levels (price and quantity). """
        out = [str(self.ct)]
        a,b = self.snap['ask'], self.snap['bid']

        for i in range(self.args.levels):
            out.extend([str(a[i].price),str(a[i].quantity)] if i < len(a) else ["nan","nan"])
            out.extend([str(b[i].price),str(b[i].quantity)] if i < len(b) else ["nan","nan"])
        return out

    def log_book(self):
        """ Return the order book snapshot waiting to be logged, if any. """
        if self.book_to_log is None: return None

        ### TEMP HACK: add fund
        to_log = self.book_to_log + self.last_trade + [self.fund]
        self.book_to_log = None
        return to_log

