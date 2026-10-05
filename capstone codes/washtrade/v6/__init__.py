r"""washtrade.v6 -- v3 mechanics with a realistic honest market and tunable camouflage.

v6 = v3 + what the detectability study needs (v1-v5 stay frozen):

  1) **Active backgrounds.** The background's mean-reversion rate ``r`` is a scenario factor.
     With the Table-3 value r = 0.01 the ZI market only trades at the start and the end of the
     run (ZIs value the asset at their forecast of the final fundamental, which is a constant for
     most of the run); r = 1e-3 keeps it trading throughout. See ``scenarios.BACKGROUNDS``.
  2) **Camouflage intensity** (``WashGroup.camouflage_lam``: own arrival rate of camouflage
     wallets, from private streams) and **partial camouflage** (``WashGroup.n_camouflage``).
  3) **Price rule per group** (``WashGroup.price_rule``: v3 midpoint or v5 microprice).
  4) **Sweeps for a cluster**: ``design.py`` (six factorial sweeps), ``generate.py`` (sharded,
     resumable, merge step), ``slurm/`` (job-array templates), ``analysis.py`` (background
     activity, camouflage share and per-group leakage).

Defaults reproduce v3 bit for bit; ``price_rule="microprice"`` reproduces v5 (``run_smoke.py``).
"""
from .simulator import PRICE_RULES, WashGroup, WashTradingSimulator

__all__ = ["PRICE_RULES", "WashGroup", "WashTradingSimulator"]
