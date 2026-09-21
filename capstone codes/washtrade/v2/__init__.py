r"""washtrade.v2 -- wash-trading mechanism, version 2.

v2 = v1 + three additions (the v1 package is frozen and left untouched):

  1) **Wallet count is a first-class knob.** 2 wallets (A<->B ping-pong) and 3 wallets
     (A->B->C->A triangle) are both supported and both exercised by the smoke test.
     n=2 is the minimal wash structure and the hardest case for a counterparty-graph
     detector (a single reciprocal edge), so Part 2 needs it as a separate condition.

  2) **The time gap between the two legs is a parameter** (``wash_gap``, in timesteps).
     v1 hard-wired gap = 0 (both wallets act in the SAME timestep and match inside one batch
     clear). v2 posts leg 1 at t, leg 2 at t + gap, so for gap > 0 the first leg sits
     EXPOSED in the book and the background traders can hit it -- this is the "exposed
     quote" mechanism v1's notes flagged as the way to actually generate spillover.
     ``wash_gap=0`` reproduces v1 exactly (verified in ``run_smoke.py``).

  3) **The limit order book is recorded and saved.** Every cleared timestep produces a
     snapshot (quotes/depth before and after the clear, clear price, volume, wash flags),
     plus optional top-K depth levels per side. ``analysis.save_run`` dumps LOB + trades +
     wash events + per-leg fills + a meta.json to disk.

Key modules:
  * wash_agent.WashTrader          -- one wallet = one agent_id = one net position.
  * controller.WashController      -- rotation + geometric timing + two-leg (gapped) events.
  * lob.LOBRecorder                -- non-invasive order-book observer.
  * simulator.WashTradingSimulator -- ComposableSimulator subclass; injects legs, records.
  * analysis                       -- post-run frames, metrics, and disk export.
"""
from .wash_agent import WashTrader
from .controller import WashController
from .lob import LOBRecorder
from .simulator import WashTradingSimulator

__all__ = ["WashTrader", "WashController", "LOBRecorder", "WashTradingSimulator"]
