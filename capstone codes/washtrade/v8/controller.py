r"""controller.py -- v8 wash controllers: v7's rounds, executed so that short time windows see less.

Detection v4 caught v7's adversary with two short-window channels:

  * **repeat**: v7 trades the Q units of an edge back to back, so the same ordered pair trades
    again within a few dozen steps;
  * **round trip**: a wallet's two legs in a round are a few events apart, so it undoes its
    position within about 100 steps.

v8 adds two execution knobs to v7's round planner. They change only the ORDER in which a
round's unit events are executed, never which events exist. Both are off by default; then v8 == v7
bit for bit.

* ``edge_order`` -- order of the unit events INSIDE a round:
      "sequential"   v7: all units of edge 1, then all of edge 2, ...
      "interleaved"  round robin over the edges: unit 1 of every edge, then unit 2, ...
                     The same pair now trades once every k events instead of Q times in a row.
* ``pipeline`` = D -- the group keeps D rounds open at once and takes its events from them in
  round-robin order. A wallet joins a new round only once its legs in the open rounds are all
  done, so a wallet never has legs planned in two open rounds; this needs D * cycle_len <= pool
  size. Between a wallet's two legs in a round lie about D other events,
  so with events at rate lam_wash it holds each position about D / lam_wash steps instead of
  1 / lam_wash. Its counterparties in the round hold the opposite position, so the GROUP stays
  flat: holding longer costs the adversary nothing in group exposure, only more wallets and a
  longer wait before rounds close.

The event itself (two legs priced off one anchor, leg 2 at t + gap, geometric timing at rate
``lam_wash``, strictly one open event per group), rotation, size, residual and every random
stream are v7's. The event times are therefore the same for every value of the two knobs. In
the round log, ``unit_idx`` is the execution order of the round's events (v7: creation order,
which is the same thing there).
"""
from collections import deque

from washtrade.v3.controller import WashController as MidController
from washtrade.v5.controller import WashController as MicropriceController
from washtrade.v7.controller import RoundPlanMixin

EDGE_ORDERS = ("sequential", "interleaved")


class PipelineMixin:
    """Execution order on top of v7's ``RoundPlanMixin``. Must come before it in the MRO."""

    def __init__(self, *args, edge_order="sequential", pipeline=1, **kwargs):
        super().__init__(*args, **kwargs)
        if edge_order not in EDGE_ORDERS:
            raise ValueError(f"edge_order must be one of {EDGE_ORDERS}, got {edge_order!r}")
        d = int(pipeline)
        if d < 1:
            raise ValueError(f"pipeline must be >= 1, got {pipeline}")
        if d > 1 and self.rotation == "fixed":
            raise ValueError("pipeline > 1 needs a rotating pool (rotation 'random' or 'lru')")
        if d * self.cycle_len > len(self.wallet_ids):
            raise ValueError(f"pipeline {d} x cycle_len {self.cycle_len} exceeds the pool "
                             f"({len(self.wallet_ids)} wallets): open rounds must be disjoint")
        if self.unit_lam is not None and (edge_order != "sequential" or d > 1):
            raise ValueError("unit_lam only applies to v7's sequential, single-round execution")
        self.edge_order = edge_order
        self.pipeline = d
        self._rounds = []                 # open rounds: deques of planned events (v8 mode only)
        self._rr = 0                      # round-robin pointer into _rounds
        self.max_open_rounds = 1          # v7 mode: one round at a time by construction

    @property
    def _v7_mode(self) -> bool:
        return self.edge_order == "sequential" and self.pipeline == 1

    # -- planning ------------------------------------------------------------ #
    def _busy(self) -> set:
        return {e[k] for r in self._rounds for e in r for k in ("seller", "buyer")}

    def _draw_members(self):
        if self._v7_mode or not self._rounds:
            return super()._draw_members()
        # a wallet is busy while it still has a planned leg in an open round; rounds are only
        # drawn between events (one open event per group), so nothing is in flight here
        busy = self._busy()
        free = [w for w in self.wallet_ids if w not in busy]
        saved = self.wallet_ids
        try:                              # v7's draw (same rng stream) over the free wallets
            self.wallet_ids = free
            return super()._draw_members()
        finally:
            self.wallet_ids = saved

    def _new_round(self):
        super()._new_round()
        if self._v7_mode:
            return
        events = list(self._plan)          # v7 queued exactly this round's events
        self._plan.clear()
        if self.edge_order == "interleaved":
            events.sort(key=lambda e: (e["piece"], e["edge_in_round"]))
        for i, e in enumerate(events):
            e["unit_idx"] = i
        self._rounds.append(deque(events))

    def _next_event(self):
        """The next planned event: keep ``pipeline`` rounds open, take them in turn."""
        while len(self._rounds) < self.pipeline:
            n = len(self._rounds)
            self._new_round()
            if len(self._rounds) == n:     # a round with no events (cannot happen with Q >= 1)
                break
        self.max_open_rounds = max(self.max_open_rounds, len(self._rounds))
        self._rr %= len(self._rounds)
        r = self._rounds[self._rr]
        ev = r.popleft()
        if r:
            self._rr += 1
        else:
            self._rounds.pop(self._rr)     # the next round slides into this slot
        return ev

    # -- event hook ---------------------------------------------------------- #
    def _post_open(self, t):
        if not self._v7_mode:
            self._plan.appendleft(self._next_event())   # v7's hook pops it
        return super()._post_open(t)


def make_controller(base):
    """A v8 controller class: ``PipelineMixin`` over v7's ``RoundPlanMixin`` over ``base``."""
    return type(f"Pipeline{base.__module__.split('.')[-2].upper()}Controller",
                (PipelineMixin, RoundPlanMixin, base), {})


PRICE_RULES = {"mid": make_controller(MidController),
               "microprice": make_controller(MicropriceController)}
