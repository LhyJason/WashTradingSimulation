r"""controller.py -- v2 WashController: two-leg wash events with a configurable TIME GAP.

Mechanism (v2): ONE directed edge per event, TWO legs separated by ``gap`` timesteps.

  * The wallets form a directed cycle A -> B -> C -> A (n = 3) or A <-> B (n = 2). Each wash
    event consumes the next edge (seller, buyer) and is executed in two legs:

        t              : leg 1 is posted  (which side goes first is ``first_leg``)
        t + gap        : leg 2 is posted  -> it crosses leg 1 in that timestep's batch clear

    Both legs are priced off ONE anchor captured at leg-1 time: the current spread MIDPOINT
    (SELL = anchor - eps, BUY = anchor + eps, default eps = 0 -> both exactly at the mid).
    Leg 2 deliberately reuses the *stale* anchor rather than re-reading the book, because
    that is what guarantees the two legs still cross each other when leg 2 arrives.

  * ``gap = 0`` -> both legs land in the SAME timestep and the same batch clear. This is
    exactly the v1 mechanism (and reproduces it bit-for-bit; see run_smoke.py check E).

  * ``gap > 0`` -> leg 1 sits EXPOSED in the book for ``gap`` timesteps. Every background
    arrival in between clears the market, so a ZI can hit the resting wash leg first. That
    is the point of the parameter: gap is the spillover knob v1 lacked (v1's note that
    ``wash_eps`` is *not* a spillover knob still holds -- keep eps = 0).

    When leg 1 is intercepted, leg 2 is still posted (the wallet pair is committed to the
    q-transfer). It then either crosses whatever the book offers, or rests unfilled and is
    withdrawn when that wallet next acts. Both outcomes are logged, never corrected away.

  * Events are strictly sequential: the next event's leg 1 is scheduled ``1 + Geom(lam_wash)``
    timesteps after the current event's leg 2, so a wallet never has two live wash orders.

The controller NEVER touches the order book itself. It hands (agent_id, Order) pairs to the
simulator, which injects them; the simulator calls ``observe()`` after EVERY cleared timestep
so that interceptions during the gap are captured.

Logs produced (the Part-1 ground-truth stream):
  * ``event_log`` -- one row per event: intent (edge, prices, leg times) + realized outcome
    (per-leg fills split into pre-close / at-close, self_trade, intercepted, ...).
  * ``fill_log``  -- one row per realized fill of a wash order (order_id level, incl. fills
    that land AFTER the event was closed).
  * ``oid_meta``  -- order_id -> which event / leg / wallet / side / price it belongs to.
"""
import math
from collections import defaultdict

import numpy as np

from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order

FIRST_LEG_MODES = ("sell", "buy", "random", "alternate")


class WashController:
    # order_ids for wash orders start high to avoid colliding with the ZI range (1..1e7).
    _OID_BASE = 1_000_000_000
    # the first-leg coin flip uses its OWN rng stream, so that changing ``first_leg`` never
    # perturbs the event-timing stream (that is what keeps gap=0 identical to v1).
    _LEG_SEED_OFFSET = 7919

    def __init__(self, wallets, market, q=1, eps=0.0, lam_wash=5e-3, gap=0,
                 first_leg="random", seed=None, sim_time=2000):
        self.wallets = list(wallets)                 # WashTrader instances
        self.wallet_ids = [w.get_id() for w in self.wallets]
        self.market = market
        self.q = q
        self.eps = eps
        self.lam_wash = lam_wash

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
        # Directed cycle edges (seller_id, buyer_id):
        #   n = 3 -> A->B, B->C, C->A   |   n = 2 -> A->B, B->A   (needs >= 2 wallets)
        self.edges = ([(self.wallet_ids[i], self.wallet_ids[(i + 1) % n])
                       for i in range(n)] if n >= 2 else [])
        self.edge_idx = 0
        self.active = n >= 2

        self.next_event_time = self._sample_gap() if self.active else None
        self._next_oid = self._OID_BASE
        self._open = None            # the event whose leg 1 is out but leg 2 is not yet closed
        self._alt = 0                # first_leg="alternate" toggle

        self.event_log = []          # one dict per completed (or dangling) event
        self.fill_log = []           # one dict per realized fill of a wash order
        self.oid_meta = {}           # order_id -> leg metadata
        self.filled_qty = defaultdict(float)   # order_id -> cumulative filled quantity

    # -- timing -------------------------------------------------------------- #
    def _sample_gap(self) -> int:
        # geometric inter-event time >= 1 (mirrors the +1 spacing used upstream for arrivals)
        return int(self.rng.geometric(self.lam_wash))

    def due(self, t) -> bool:
        """True if this timestep must be stepped for a wash leg (leg 1 OR leg 2)."""
        if not self.active:
            return False
        if self._open is not None:
            return t == self._open["close_time"]
        return self.next_event_time is not None and t == self.next_event_time

    def pending(self) -> bool:
        """True while an event has leg 1 out and leg 2 still to come."""
        return self._open is not None

    # -- order construction -------------------------------------------------- #
    def _new_oid(self) -> int:
        oid = self._next_oid
        self._next_oid += 1
        return oid

    def _anchor_and_delta(self):
        """Anchor the crossing bracket at the current spread MIDPOINT, then offset each leg by
        ``delta = eps`` (SELL = anchor-eps, BUY = anchor+eps).

        Default eps = 0: both legs sit exactly at the midpoint, strictly inside the spread.
        With gap = 0 no resting order can reach them, so the pair can only match EACH OTHER
        and the trade prints at the midpoint (~zero price impact).

        With gap > 0 the midpoint anchor is what makes the exposed leg *attractive but not
        aggressive*: it becomes the new best quote on its side, so a ZI can take it, but it
        does not itself cross anything on arrival.

        NOTE on eps > 0 (kept as a parameter, but NOT a spillover knob -- v1 finding): pushing
        SELL down and BUY up only makes the pair more dominant on both sides; its only real
        effect is to drag the uniform clearing price to anchor-eps (a downward price impact),
        which contradicts the pure-volume / no-manipulation scope. Leave eps at 0 and use
        ``gap`` to generate spillover instead.
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

        return anchor, max(0.0, self.eps), bb, ba

    def _pick_first_leg(self) -> int:
        """Which side is exposed first. Irrelevant when gap = 0 (both legs clear together)."""
        if self.first_leg == "sell":
            return SELL
        if self.first_leg == "buy":
            return BUY
        if self.first_leg == "alternate":
            self._alt ^= 1
            return SELL if self._alt else BUY
        return SELL if self._leg_rng.random() < 0.5 else BUY

    def _make_leg(self, ev, side, t):
        """Build one leg of ``ev`` at time t, register its order_id, return (agent_id, Order)."""
        if side == SELL:
            aid, px, key = ev["seller"], ev["sell_px"], "sell_oid"
        else:
            aid, px, key = ev["buyer"], ev["buy_px"], "buy_oid"
        oid = self._new_oid()
        ev[key] = oid
        order = Order(price=px, order_type=side, quantity=self.q,
                      agent_id=aid, time=t, order_id=oid)
        self.oid_meta[oid] = dict(
            event_idx=ev["event_idx"], agent_id=aid,
            side=("SELL" if side == SELL else "BUY"),
            role=("seller" if side == SELL else "buyer"),
            leg=(1 if side == ev["_first_side"] else 2),
            post_time=t, price=float(px), q=self.q,
        )
        return aid, order

    def orders_for(self, t):
        """Wash orders to inject at time t: a list of (agent_id, Order).

        The simulator must call this AFTER ``market.event_queue.set_time(t)`` (leg 1 reads the
        fundamental at t) and must ``withdraw_all`` each posting agent before adding its order.
        """
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
        # At gap = 0 the two legs land in the same batch clear, so "which one is first" has no
        # meaning; we fix the convention SELL-then-BUY (leg 1 = SELL) and skip the coin flip.
        # That also makes gap=0 byte-identical to v1: same order_id assignment and the same
        # insertion order into the event queue.
        first = SELL if self.gap == 0 else self._pick_first_leg()

        ev = dict(
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
            # v1 behaviour: both legs in the same timestep -> same batch clear.
            posts.append(self._make_leg(ev, -first, t))
            ev["_close_posted"] = True
        return posts

    def _post_close(self, t):
        ev = self._open
        ev["_close_posted"] = True
        return [self._make_leg(ev, -ev["_first_side"], t)]

    # -- realized-fill logging ---------------------------------------------- #
    def observe(self, t, new_matched):
        """Called by the simulator after EVERY cleared timestep.

        ``new_matched`` is the list of MatchedOrder produced by this timestep's clear. Fills of
        wash orders are accumulated at order_id level (so an interception during the gap is
        attributed to the exact leg it hit), and the open event is finalized on its close step.
        """
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
                time=t, order_id=oid, event_idx=meta["event_idx"], leg=meta["leg"],
                role=meta["role"], agent_id=meta["agent_id"], side=meta["side"],
                qty=qty, price=float(mo.price), post_time=meta["post_time"],
                lag=t - meta["post_time"],
                # a fill on an event that was already finalized = a leftover leg filled late
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
            # fills BEFORE the close step: only the exposed leg can have any -> interception
            seller_filled_pre=s_tot - s_close, buyer_filled_pre=b_tot - b_close,
            seller_filled_close=s_close, buyer_filled_close=b_close,
            # total realized as of the close step (what the booked positions reflect)
            seller_filled=s_tot, buyer_filled=b_tot,
            # both intended legs crossed in the SAME clear -> a genuine self-trade
            self_trade=bool(s_close >= self.q and b_close >= self.q),
            # the exposed leg was (partly) taken by somebody else before the partner arrived
            intercepted=bool((s_tot - s_close) > 0 or (b_tot - b_close) > 0),
            # the pair failed to move the full q (leg 2 rested unfilled, or partial fills)
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
        """End of run: close out an event whose leg 2 never got a timestep (gap ran off the
        end of the simulation). Logged with ``closed=False`` so it is never counted as a
        successful wash event."""
        if self._open is not None:
            self._finalize(self._open["close_time"], {}, 0, 0, closed=False)
