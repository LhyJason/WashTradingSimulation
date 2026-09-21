r"""controller.py -- WashController: coordinates a set of wallets that trade with each other.

Mechanism (v1): ACTIVE co-arrival, ONE directed edge per event.

  * The wallets form a directed cycle A -> B -> C -> A. Each wash event consumes the next
    edge (seller, buyer): the seller and buyer both act in the SAME timestep, posting a
    crossing pair anchored at the current spread MIDPOINT (SELL = mid - eps, BUY = mid + eps;
    default eps = 0 -> both at the midpoint) so that
      (a) they cross each other (buy price >= sell price), and
      (b) both legs sit strictly INSIDE the current bid-ask spread, out of reach of every
          resting order, so the pair can only match each other, and
      (c) the clear prints at the midpoint -> ~zero price impact (pure volume, no manipulation).
    After one full cycle of n events every wallet is back to net-zero:
      A->B: A -q, B +q ; B->C: B -q, C +q ; C->A: C -q, A +q  =>  each wallet 0.

  * Event timing is geometric with rate ``lam_wash``, independent of and seeded separately
    from the ZI background, so wash intensity is a clean experimental knob.

The controller NEVER enters the order book itself. It only builds the two Order objects that
the simulator injects; the simulator's batch clear matches them, and the upstream
matched-order loop books the positions. Interceptions by a ZI ("spillover") are recorded in
``event_log`` (seller_filled / buyer_filled / self_trade), not corrected away.
"""
import math

import numpy as np

from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order


class WashController:
    # order_ids for wash orders start high to avoid colliding with the ZI range (1..1e7).
    _OID_BASE = 1_000_000_000

    def __init__(self, wallets, market, q=1, eps=0.0, lam_wash=5e-3,
                 seed=None, sim_time=2000):
        self.wallets = list(wallets)                 # WashTrader instances
        self.wallet_ids = [w.get_id() for w in self.wallets]
        self.market = market
        self.q = q
        self.eps = eps
        self.lam_wash = lam_wash
        self.sim_time = sim_time
        self.rng = np.random.default_rng(seed)

        n = len(self.wallets)
        # Directed cycle edges (seller_id, buyer_id): A->B, B->C, C->A. Needs >= 2 wallets.
        self.edges = ([(self.wallet_ids[i], self.wallet_ids[(i + 1) % n])
                       for i in range(n)] if n >= 2 else [])
        self.edge_idx = 0
        self.active = n >= 2

        self.next_event_time = self._sample_gap() if self.active else None
        self._next_oid = self._OID_BASE
        self._pending = None
        self.event_log = []          # one dict per fired event

    # -- timing -------------------------------------------------------------- #
    def _sample_gap(self) -> int:
        # geometric inter-event time >= 1 (mirrors the +1 spacing used upstream for arrivals)
        return int(self.rng.geometric(self.lam_wash))

    def due(self, t) -> bool:
        return self.active and self.next_event_time is not None and t == self.next_event_time

    def advance(self):
        self.edge_idx = (self.edge_idx + 1) % len(self.edges)
        self.next_event_time = self.next_event_time + 1 + self._sample_gap()

    # -- order construction -------------------------------------------------- #
    def _new_oid(self) -> int:
        oid = self._next_oid
        self._next_oid += 1
        return oid

    def _anchor_and_delta(self):
        """Anchor the crossing bracket at the current spread MIDPOINT, then offset each leg by
        ``delta = eps`` (SELL = anchor-eps, BUY = anchor+eps).

        Default is eps = 0: both legs sit exactly at the midpoint, strictly inside the spread.
        No resting order can reach them, so the pair can only match EACH OTHER -- internal wash
        volume, net-zero, and the trade prints at the midpoint (~zero price impact). Empirically
        this self-trades ~100% of the time and is insensitive to seed.

        NOTE on eps > 0 (kept as a parameter, but NOT a spillover knob): pushing SELL down and
        BUY up only makes the pair *more* dominant on both sides, so it still self-trades every
        time -- it does not induce ZI interception. Its only real effect is to move the uniform
        clearing price to anchor-eps (the bid quote becomes the wash sell), i.e. it adds a
        downward PRICE IMPACT. Since v1's scope is pure volume with no price manipulation, leave
        eps at 0. Deliberately generating spillover needs a different mechanism (a less dominant,
        exposed quote), which is out of scope for v1.
        """
        ob = self.market.order_book
        bb = ob.get_best_bid()     # highest resting buy  (-inf if none)
        ba = ob.get_best_ask()     # lowest  resting sell (+inf if none)
        f = float(self.market.get_fundamental_value())

        if math.isinf(bb) and math.isinf(ba):
            anchor = f                       # empty book: fall back to the fundamental
        elif math.isinf(ba):                 # only bids rest -> stay above the best bid
            anchor = max(f, bb + 1.0)
        elif math.isinf(bb):                 # only asks rest -> stay below the best ask
            anchor = min(f, ba - 1.0)
        else:                                # both sides -> midpoint of the spread
            anchor = 0.5 * (bb + ba)

        return anchor, max(0.0, self.eps)

    def make_wash_pair(self, t):
        """Build the crossing (SELL, BUY) pair for the current edge at time t.
        Returns (seller_id, buyer_id, sell_order, buy_order)."""
        seller_id, buyer_id = self.edges[self.edge_idx]
        anchor, delta = self._anchor_and_delta()
        sell_px = anchor - delta
        buy_px = anchor + delta

        sell_order = Order(price=sell_px, order_type=SELL, quantity=self.q,
                           agent_id=seller_id, time=t, order_id=self._new_oid())
        buy_order = Order(price=buy_px, order_type=BUY, quantity=self.q,
                          agent_id=buyer_id, time=t, order_id=self._new_oid())

        self._pending = dict(t=t, edge_idx=self.edge_idx, seller=seller_id, buyer=buyer_id,
                             q=self.q, anchor=anchor, sell_px=sell_px, buy_px=buy_px)
        return seller_id, buyer_id, sell_order, buy_order

    # -- realized-fill logging ---------------------------------------------- #
    def record(self, t, new_matched):
        """Log realized fills for the event just cleared.
        ``new_matched`` is the list of MatchedOrder produced by this timestep's clear."""
        p = dict(self._pending)
        seller, buyer = p["seller"], p["buyer"]
        seller_filled = sum(mo.order.quantity for mo in new_matched
                            if mo.order.agent_id == seller and mo.order.order_type == SELL)
        buyer_filled = sum(mo.order.quantity for mo in new_matched
                           if mo.order.agent_id == buyer and mo.order.order_type == BUY)
        p["seller_filled"] = seller_filled
        p["buyer_filled"] = buyer_filled
        # self_trade: both legs of the intended pair filled at least q this step. (A ZI could
        # still have taken one leg while the other rested; we flag partials via the counts.)
        p["self_trade"] = bool(seller_filled >= self.q and buyer_filled >= self.q)
        p["n_matched_this_step"] = len(new_matched)
        self.event_log.append(p)
        self._pending = None
