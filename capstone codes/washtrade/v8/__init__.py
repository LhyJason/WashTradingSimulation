r"""washtrade.v8 -- v7 with an adversary that hides from short time windows, and multi-unit
honest orders (v1-v7 stay frozen).

  1) **Edge order** (``WashGroup.edge_order``): "interleaved" trades a round's units round robin
     over its edges, so the same pair no longer trades several times in a row.
  2) **Pipeline** (``WashGroup.pipeline`` = D): D rounds open at once, events taken in turn, so
     each wallet holds a position about D times longer while the group stays flat.
  3) **Multi-unit honest orders** (simulator / background ``order_size_max`` = m): ZI traders
     post orders of 1..m units (``agents.MultiUnitZI``), so honest positions are no longer +-1
     walks and a multi-unit wash order no longer gives itself away.

Defaults reproduce v7 (and so v6, v3) bit for bit; see ``run_smoke.py``.
"""
from .agents import MultiUnitZI
from .controller import EDGE_ORDERS, PRICE_RULES
from .simulator import WashGroup, WashTradingSimulator

__all__ = ["EDGE_ORDERS", "MultiUnitZI", "PRICE_RULES", "WashGroup", "WashTradingSimulator"]
