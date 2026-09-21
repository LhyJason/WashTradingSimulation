r"""washtrade.v3 -- wash-trading mechanism, version 3 (the data generator for Part 2).

v3 = v2 + what Part 2 needs to be a real test rather than a give-away (v1/v2 stay frozen):

  1) **Camouflage wallets** (``WashGroup(camouflage=True)``). In v1/v2 a wash wallet ONLY ever
     wash-traded, so "never trades normally" alone identified it. A camouflage wallet is a
     regular ZI trader (own private values, own geometric arrivals) that is ALSO enrolled in a
     wash group -- the "commingling with authentic orders" pattern of Sirolly et al. (2026),
     Example 7. Supporting it needs order-scoped cancellation: upstream ``withdraw_all`` would
     let the wallet's ZI arrival cancel its own exposed wash leg.

  2) **Several independent wash groups** in one market (``wash_groups=[...]``), each with its own
     controller, wallet count, gap, intensity and camouflage flag. Opens of different groups
     never share a timestep (a colliding open is deferred by one step), so groups do not trade
     into each other's crossing pairs by accident.

  3) **A fixed export schema** that separates what a detector may see from the labels:
       observable/  trades.csv (anonymized wallets, no order ids), lob.csv, lob_levels.csv
       labels/      wallets.csv, trade_legs.csv, wash_events.csv, wash_fills.csv,
                    wash_orders.csv, lob_latent.csv
     plus ``scenarios.py`` / ``generate.py`` to produce a whole labelled dataset in one command.
     TODO: the schema is a working default; see the note in ``export.py``.

With a single group and ``camouflage=False`` v3 reproduces v2 exactly (``run_smoke.py`` check B).
"""
from .wash_agent import WashTrader
from .controller import WashController
from .lob import LOBRecorder
from .simulator import WashGroup, WashTradingSimulator

__all__ = ["WashTrader", "WashController", "LOBRecorder", "WashGroup", "WashTradingSimulator"]
