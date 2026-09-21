r"""controller.py -- v3 WashController: one wash GROUP's two-leg (optionally gapped) events.

Mechanism is unchanged from v2 (see washtrade/v2/controller.py for the full rationale):

  * wallets form a directed cycle (n = 2: A <-> B, n = 3: A -> B -> C -> A, ...); each event
    trades the next edge (seller, buyer) with two legs priced off ONE midpoint anchor captured
    at leg-1 time (SELL = anchor - eps, BUY = anchor + eps; eps = 0 by default);
  * leg 1 at t, leg 2 at t + gap. gap = 0 -> both legs in the same batch clear (v1 behaviour);
    gap > 0 -> leg 1 is exposed and can be intercepted (the spillover knob);
  * events are strictly sequential within a group; timing is geometric with rate ``lam_wash``.

What v3 adds for multi-group runs:

  * ``group_id`` -- stamped on every event / fill / order row;
  * a DISJOINT order_id range per group (base 1e9 + group_id * 1e8), so any wash order can be
    attributed to its group from the id alone, and every wash id is >= 1e9 while background
    (ZI) ids are <= 1e7. The simulator relies on that split for order-scoped cancellation;
  * ``defer_open()`` -- lets the simulator push this group's next event start by one timestep
    when another group already posts in that timestep.

Group 0 keeps exactly v2's order_id base and rng streams, which is what lets a single-group v3
run reproduce v2 bit-for-bit.
"""
import math
from collections import defaultdict

import numpy as np

from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order

FIRST_LEG_MODES = ("sell", "buy", "random", "alternate")

# every wash order_id is >= this; background agents draw ids from 1..1e7
WASH_OID_FLOOR = 1_000_000_000


def is_wash_oid(order_id) -> bool:
    return int(order_id) >= WASH_OID_FLOOR


class WashController:
    _OID_BASE = WASH_OID_FLOOR
    _OID_GROUP_STRIDE = 100_000_000
    # the first-leg coin flip uses its OWN rng stream, so changing ``first_leg`` never
    # perturbs the event-timing stream.
    _LEG_SEED_OFFSET = 7919

    def __init__(self, wallets, market, q=1, eps=0.0, lam_wash=5e-3, gap=0,
                 first_leg="random", seed=None, sim_time=2000, group_id=0):
        self.wallets = list(wallets)
        self.wallet_ids = [w.get_id() for w in self.wallets]
        self.market = market
        self.q = q
        self.eps = eps
        self.lam_wash = lam_wash
        self.group_id = int(group_id)

        gap = int(gap)
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        self.gap = gap

        first_leg = str(first_leg).lower()
        if first_leg not in FIRST_LEG_MODES:
            raise ValueError(f"first_leg must be one of {FIRST_LEG_MODES}, got {first_leg!r}")
        self.first_leg = first_leg

        self.sim_time = sim_time
        self.rng = np.random.default_rng(seed)
        self._leg_rng = np.random.default_rng(
            None if seed is None else int(seed) + self._LEG_SEED_OFFSET)

        n = len(self.wallets)
        self.edges = ([(self.wallet_ids[i], self.wallet_ids[(i + 1) % n])
                       for i in range(n)] if n >= 2 else [])
        self.edge_idx = 0
        self.active = n >= 2

        self.next_event_time = self._sample_gap() if self.active else None
        self._next_oid = self._OID_BASE + self.group_id * self._OID_GROUP_STRIDE
        self._open = None
        self._alt = 0
        self.n_deferred = 0

        self.event_log = []
        self.fill_log = []
        self.oid_meta = {}
        self.filled_qty = defaultdict(float)

    # -- timing -------------------------------------------------------------- #
    def _sample_gap(self) -> int:
        return int(self.rng.geometric(self.lam_wash))

    def due(self, t) -> bool:
        """True if this timestep must be stepped for one of this group's legs."""
        if not self.active:
            return False
        if self._open is not None:
            return t == self._open["close_time"]
        return self.next_event_time is not None and t == self.next_event_time

    def pending(self) -> bool:
        """True while an event has leg 1 out and leg 2 still to come."""
        return self._open is not None

    def defer_open(self):
        """Push the next event start back one timestep (called on a cross-group collision)."""
        if self._open is None and self.next_event_time is not None:
            self.next_event_time += 1
            self.n_deferred += 1

    # -- order construction -------------------------------------------------- #
    def _new_oid(self) -> int:
        oid = self._next_oid
        self._next_oid += 1
        return oid

    def _anchor_and_delta(self):
        """Midpoint anchor (fundamental fallback on a one-sided / empty book). See v2 docstring
        for why eps is not a spillover knob and should stay 0."""
        ob = self.market.order_book
        bb = ob.get_best_bid()
        ba = ob.get_best_ask()
        f = float(self.market.get_fundamental_value())

        if math.isinf(bb) and math.isinf(ba):
            anchor = f
        elif math.isinf(ba):
            anchor = max(f, bb + 1.0)
        elif math.isinf(bb):
            anchor = min(f, ba - 1.0)
        else:
            anchor = 0.5 * (bb + ba)

        return anchor, max(0.0, self.eps), bb, ba

    def _pick_first_leg(self) -> int:
        if self.first_leg == "sell":
            return SELL
        if self.first_leg == "buy":
            return BUY
        if self.first_leg == "alternate":
            self._alt ^= 1
            return SELL if self._alt else BUY
        return SELL if self._leg_rng.random() < 0.5 else BUY

    def _make_leg(self, ev, side, t):
        if side == SELL:
            aid, px, key = ev["seller"], ev["sell_px"], "sell_oid"
        else:
            aid, px, key = ev["buyer"], ev["buy_px"], "buy_oid"
        oid = self._new_oid()
        ev[key] = oid
        order = Order(price=px, order_type=side, quantity=self.q,
                      agent_id=aid, time=t, order_id=oid)
        self.oid_meta[oid] = dict(
            group_id=self.group_id, event_idx=ev["event_idx"], agent_id=aid,
            side=("SELL" if side == SELL else "BUY"),
            role=("seller" if side == SELL else "buyer"),
            leg=(1 if side == ev["_first_side"] else 2),
            post_time=t, price=float(px), q=self.q,
        )
        return aid, order

    def orders_for(self, t):
        """Wash orders to inject at time t: a list of (agent_id, Order). Call after
        ``market.event_queue.set_time(t)``."""
        if not self.active or t >= self.sim_time:
            return []
        if self._open is not None:
            if t != self._open["close_time"] or self._open.get("_close_posted"):
                return []
            return self._post_close(t)
        if self.next_event_time is None or t != self.next_event_time:
            return []
        return self._post_open(t)

    def _post_open(self, t):
        seller_id, buyer_id = self.edges[self.edge_idx]
        anchor, delta, bb, ba = self._anchor_and_delta()
        # gap = 0: fixed SELL-then-BUY convention, no coin flip (keeps v1/v2 order_id and
        # event-queue insertion order).
        first = SELL if self.gap == 0 else self._pick_first_leg()

        ev = dict(
            group_id=self.group_id,
            event_idx=len(self.event_log), edge_idx=self.edge_idx,
            seller=seller_id, buyer=buyer_id, q=self.q, gap=self.gap,
            open_time=t, close_time=t + self.gap,
            first_leg=("simultaneous" if self.gap == 0
                       else ("SELL" if first == SELL else "BUY")),
            anchor=float(anchor), sell_px=float(anchor - delta), buy_px=float(anchor + delta),
            best_bid_at_open=float(bb), best_ask_at_open=float(ba),
            _first_side=first,
        )
        self._open = ev

        posts = [self._make_leg(ev, first, t)]
        if self.gap == 0:
            posts.append(self._make_leg(ev, -first, t))
            ev["_close_posted"] = True
        return posts

    def _post_close(self, t):
        ev = self._open
        ev["_close_posted"] = True
        return [self._make_leg(ev, -ev["_first_side"], t)]

    # -- realized-fill logging ---------------------------------------------- #
    def observe(self, t, new_matched):
        """Called after EVERY cleared timestep. Only this group's order_ids are attributed;
        legs of other groups count as outsiders."""
        step_fills = defaultdict(float)
        outsider_legs = 0
        for mo in new_matched:
            oid = mo.order.order_id
            meta = self.oid_meta.get(oid)
            if meta is None:
                outsider_legs += 1
                continue
            qty = float(mo.order.quantity)
            step_fills[oid] += qty
            self.filled_qty[oid] += qty
            self.fill_log.append(dict(
                time=t, group_id=self.group_id, order_id=oid, event_idx=meta["event_idx"],
                leg=meta["leg"], role=meta["role"], agent_id=meta["agent_id"],
                side=meta["side"], qty=qty, price=float(mo.price), post_time=meta["post_time"],
                lag=t - meta["post_time"],
                late=bool(meta["event_idx"] < len(self.event_log)),
            ))

        if self._open is not None and t == self._open["close_time"]:
            self._finalize(t, step_fills, outsider_legs, len(new_matched), closed=True)

    def _finalize(self, t, step_fills, outsider_legs, n_matched, closed):
        ev = self._open
        oid_s = ev.get("sell_oid")
        oid_b = ev.get("buy_oid")

        s_close = float(step_fills.get(oid_s, 0.0)) if oid_s is not None else 0.0
        b_close = float(step_fills.get(oid_b, 0.0)) if oid_b is not None else 0.0
        s_tot = float(self.filled_qty.get(oid_s, 0.0)) if oid_s is not None else 0.0
        b_tot = float(self.filled_qty.get(oid_b, 0.0)) if oid_b is not None else 0.0

        ev.update(
            seller_filled_pre=s_tot - s_close, buyer_filled_pre=b_tot - b_close,
            seller_filled_close=s_close, buyer_filled_close=b_close,
            seller_filled=s_tot, buyer_filled=b_tot,
            self_trade=bool(s_close >= self.q and b_close >= self.q),
            intercepted=bool((s_tot - s_close) > 0 or (b_tot - b_close) > 0),
            unfilled=float(2 * self.q - s_tot - b_tot),
            outsider_legs_at_close=outsider_legs,
            n_matched_this_step=n_matched,
            closed=bool(closed),
        )
        ev.pop("_first_side", None)
        ev.pop("_close_posted", None)
        self.event_log.append(ev)
        self._open = None
        self.edge_idx = (self.edge_idx + 1) % len(self.edges)
        self.next_event_time = t + 1 + self._sample_gap()

    def finalize(self):
        """End of run: close out an event whose leg 2 never got a timestep (``closed=False``)."""
        if self._open is not None:
            self._finalize(self._open["close_time"], {}, 0, 0, closed=False)
