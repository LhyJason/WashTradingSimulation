r"""washtrade.v7 -- v6 with an evasive wash adversary (v1-v6 stay frozen).

A group's wallets become a POOL; each round trades a closed cycle over wallets drawn from it:

  1) **Rotation** (``WashGroup.rotation``, ``cycle_len``): each round draws its k members from
     the pool ("random" or "lru"), so the same pair rarely trades twice.
  2) **Random size** (``q_range``, ``size_exec``): round size Q, executed as Q unit events by
     default (the ZI background only ever trades one unit, so qty > 1 legs would give wash away).
  3) **Residual** (``residual_prob``, ``residual_max``): the closing edge is off by a small
     delta, so individual wallets stop returning exactly to zero.

Defaults reproduce v6 (and so v3) bit for bit; see ``run_smoke.py``.
"""
from .controller import PRICE_RULES, ROTATIONS, SIZE_EXECS
from .simulator import WashGroup, WashTradingSimulator

__all__ = ["PRICE_RULES", "ROTATIONS", "SIZE_EXECS", "WashGroup", "WashTradingSimulator"]
