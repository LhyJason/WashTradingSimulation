r"""controller.py -- v7 wash controllers: rounds drawn from a wallet pool, random size, residuals.

v3's controller trades a FIXED directed cycle over the group's wallets (A -> B -> C -> A -> ...),
one unit per event, so the group returns to all-zero positions after every n events and the same
few pairs trade again and again. v7 keeps v3's EVENT (two legs priced off one anchor, leg 2 at
t + gap, strictly sequential, geometric timing) and changes what the events are used for.

A **round** is one closed cycle over ``cycle_len`` = k wallets drawn from the group's wallet pool
(``n_wallets`` = P wallets):

    members  w_1 .. w_k                 drawn per round (``rotation``)
    edges    w_1 -> w_2, ..., w_k -> w_1 (seller -> buyer)
    size     Q units on every edge       drawn per round (``q_range``)
    residual the closing edge w_k -> w_1 trades Q + delta units instead of Q
             (``residual_prob``, ``residual_max``) -> w_1 ends the round at +delta, w_k at -delta

Knobs (all off by default; then v7 == v6 bit for bit):

* ``rotation``  "fixed"   v3: k = P, members in pool order every round (no draw)
                "random"  k distinct wallets drawn uniformly from the pool, in random order
                "lru"     the k least-used wallets (ties broken at random), in random order
* ``q_range``   None = every round has Q = 1; (lo, hi) = Q uniform on {lo, ..., hi}
* ``size_exec`` "units"  an edge of size Q is Q consecutive unit events (every tape leg has
                         qty = base q, like the ZI background, whose orders are all 1 unit)
                "block"  one event with quantity Q (legs with qty > 1: no ZI ever trades more
                         than one unit, so this is visible on the tape; kept as a contrast)
* ``unit_lam``  rate of the wait between two unit events of the SAME edge; None = ``lam_wash``
* ``residual_prob`` / ``residual_max``  with probability p a round's closing edge trades
                Q + delta, |delta| uniform on {1..d}, random sign; delta is flipped (or set to 0)
                if it would push either wallet's intended inventory outside [-d, d], so residual
                positions stay small. Q + delta = 0 drops the closing edge.

The group as a whole still nets to zero (its wallets only trade with each other); residuals
only stop INDIVIDUAL wallets from returning exactly to zero. Draws come from three private
streams (rotation, size, residual) seeded off the group seed, so the event-timing stream is the
same as v3's whatever the knobs, and changing one knob does not reshuffle the others.

``make_controller(base)`` puts the round logic on top of any v3-compatible controller (v3 mid,
v5 microprice), the same way v6 chooses a price rule.
"""
from collections import deque

import numpy as np

from washtrade.v3.controller import WashController as MidController
from washtrade.v5.controller import WashController as MicropriceController

ROTATIONS = ("fixed", "random", "lru")
SIZE_EXECS = ("units", "block")


class RoundPlanMixin:
    """Round planning on top of a v3-style controller. Must come first in the MRO."""
    _ROT_SEED_OFFSET = 70_001
    _SIZE_SEED_OFFSET = 70_003
    _RES_SEED_OFFSET = 70_009
    _UNIT_SEED_OFFSET = 70_013

    def __init__(self, *args, cycle_len=None, rotation="fixed", q_range=None, size_exec="units",
                 unit_lam=None, residual_prob=0.0, residual_max=1, **kwargs):
        seed = kwargs.get("seed")
        super().__init__(*args, **kwargs)
        n = len(self.wallets)
        k = n if cycle_len is None else int(cycle_len)
        if not 2 <= k <= n:
            raise ValueError(f"cycle_len must be in [2, n_wallets={n}], got {k}")
        if rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}, got {rotation!r}")
        if rotation == "fixed" and k != n:
            raise ValueError("rotation='fixed' trades the whole pool: needs cycle_len == n_wallets")
        if size_exec not in SIZE_EXECS:
            raise ValueError(f"size_exec must be one of {SIZE_EXECS}, got {size_exec!r}")
        if q_range is not None:
            lo, hi = (int(v) for v in q_range)
            if not 1 <= lo <= hi:
                raise ValueError(f"q_range must satisfy 1 <= lo <= hi, got {q_range}")
            q_range = (lo, hi)
        if not 0.0 <= residual_prob <= 1.0:
            raise ValueError(f"residual_prob must be in [0, 1], got {residual_prob}")
        if int(residual_max) < 1:
            raise ValueError(f"residual_max must be >= 1, got {residual_max}")
        if unit_lam is not None and not 0 < unit_lam <= 1:
            raise ValueError(f"unit_lam must be in (0, 1], got {unit_lam}")

        self.cycle_len = k
        self.rotation = rotation
        self.q_range = q_range
        self.size_exec = size_exec
        self.unit_lam = unit_lam
        self.residual_prob = float(residual_prob)
        self.residual_max = int(residual_max)
        self.base_q = self.q                      # quantity of one unit event

        def _rng(off):
            return np.random.default_rng(None if seed is None else int(seed) + off)
        self._rot_rng = _rng(self._ROT_SEED_OFFSET)
        self._size_rng = _rng(self._SIZE_SEED_OFFSET)
        self._res_rng = _rng(self._RES_SEED_OFFSET)
        self._unit_rng = _rng(self._UNIT_SEED_OFFSET)

        self._plan = deque()                      # planned events of the current round
        self._cur = None                          # plan entry of the open event
        self.round_log = []
        self.inventory = {w: 0 for w in self.wallet_ids}   # intended (planned) wash position
        self.rounds_used = {w: 0 for w in self.wallet_ids}

    # -- planning ------------------------------------------------------------ #
    def _draw_members(self):
        if self.rotation == "fixed":
            return list(self.wallet_ids)
        k = self.cycle_len
        if self.rotation == "random":
            idx = self._rot_rng.choice(len(self.wallet_ids), size=k, replace=False)
            return [self.wallet_ids[int(i)] for i in idx]
        tie = self._rot_rng.random(len(self.wallet_ids))
        order = sorted(range(len(self.wallet_ids)),
                       key=lambda i: (self.rounds_used[self.wallet_ids[i]], tie[i]))
        chosen = [self.wallet_ids[i] for i in order[:k]]
        return [chosen[int(i)] for i in self._rot_rng.permutation(k)]

    def _draw_size(self) -> int:
        if self.q_range is None:
            return 1
        lo, hi = self.q_range
        return int(self._size_rng.integers(lo, hi + 1))

    def _draw_residual(self, first, last, size) -> int:
        if self.residual_prob <= 0.0 or self.cycle_len < 2:
            return 0
        if self._res_rng.random() >= self.residual_prob:
            return 0
        d = self.residual_max
        mag = int(self._res_rng.integers(1, d + 1))
        sign = 1 if self._res_rng.random() < 0.5 else -1
        for s in (sign, -sign):
            delta = max(s * mag, -size)            # the closing edge cannot trade < 0 units
            if (abs(self.inventory[first] + delta) <= d
                    and abs(self.inventory[last] - delta) <= d):
                return delta
        return 0

    def _new_round(self):
        members = self._draw_members()
        size = self._draw_size()
        k = len(members)
        delta = self._draw_residual(members[0], members[-1], size)
        r = len(self.round_log)
        edges = [(members[i], members[(i + 1) % k]) for i in range(k)]
        sizes = [size] * k
        sizes[-1] = size + delta
        unit_idx = 0
        for e, ((seller, buyer), units) in enumerate(zip(edges, sizes)):
            if units <= 0:
                continue
            pieces = [self.base_q] * units if self.size_exec == "units" else [self.base_q * units]
            for p, q in enumerate(pieces):
                self._plan.append(dict(seller=seller, buyer=buyer, q=q, round_idx=r,
                                       unit_idx=unit_idx, edge_in_round=e, piece=p,
                                       last_piece=(p == len(pieces) - 1)))
                unit_idx += 1
        for w in members:
            self.rounds_used[w] += 1
        self.inventory[members[0]] += delta * self.base_q
        self.inventory[members[-1]] -= delta * self.base_q
        self.round_log.append(dict(
            group_id=self.group_id, round_idx=r, members=list(members), cycle_len=k,
            size=size, residual=delta, n_events=unit_idx,
            planned_volume=float(sum(sizes) * self.base_q),
        ))

    # -- event hooks --------------------------------------------------------- #
    def _post_open(self, t):
        if not self._plan:
            self._new_round()
        cur = self._plan.popleft()
        self._cur = cur
        self.edges = [(cur["seller"], cur["buyer"])]
        self.edge_idx = 0
        self.q = cur["q"]
        posts = super()._post_open(t)
        self._open.update(edge_idx=cur["edge_in_round"], round_idx=cur["round_idx"],
                          unit_idx=cur["unit_idx"], piece=cur["piece"])
        rl = self.round_log[cur["round_idx"]]
        rl.setdefault("open_time", t)
        return posts

    def _finalize(self, t, step_fills, outsider_legs, n_matched, closed):
        cur = self._cur
        super()._finalize(t, step_fills, outsider_legs, n_matched, closed)
        self.round_log[cur["round_idx"]]["close_time"] = t
        # next piece of the same edge: optional own (faster) rate; v3's timing draw above is
        # still made, so the timing stream stays aligned with v3
        if (self.unit_lam is not None and not cur["last_piece"] and self._plan
                and self.next_event_time is not None):
            self.next_event_time = t + 1 + int(self._unit_rng.geometric(self.unit_lam))
        self._cur = None


def make_controller(base):
    """A v7 controller class: ``RoundPlanMixin`` over ``base`` (a v3-compatible controller)."""
    return type(f"Round{base.__module__.split('.')[-2].upper()}Controller",
                (RoundPlanMixin, base), {})


PRICE_RULES = {"mid": make_controller(MidController),
               "microprice": make_controller(MicropriceController)}
