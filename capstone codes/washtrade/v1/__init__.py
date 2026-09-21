r"""washtrade.v1 -- wash-trading mechanism, version 1.

Design:
  * Pairing mechanism   : ACTIVE co-arrival. A shared controller schedules a pair of wallets
                          into the SAME timestep and posts tightly-crossing orders, so they
                          match each other in that timestep's single batch clear.
  * Event granularity   : ONE directed edge per wash event. The wallets form a directed
                          cycle A->B->C->A; each event trades one edge. After one full cycle
                          (n events) every wallet is back to net-zero.
  * Net-zero enforcement: statistical (no hard correction). Legs almost always fill because
                          the pair dominates the top of book; the rare interception by a ZI
                          ("spillover") is logged, not hidden -- it is a phenomenon we want
                          to observe, not engineer away.

Key modules:
  * wash_agent.WashTrader          -- one wallet = one agent_id = one net position.
  * controller.WashController      -- rotation + geometric timing + builds the crossing pair.
  * simulator.WashTradingSimulator -- ComposableSimulator subclass; injects the pair and
                                      triggers wash events, reusing the upstream batch clear.
"""
from .wash_agent import WashTrader
from .controller import WashController
from .simulator import WashTradingSimulator

__all__ = ["WashTrader", "WashController", "WashTradingSimulator"]
