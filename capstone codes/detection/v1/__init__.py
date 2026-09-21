r"""detection.v1 -- network-based wash-trade detection of Sirolly, Ma, Kanoria & Sethi (2026),
"Network-Based Detection of Wash Trading", applied to the washtrade.v3 dataset.

The paper's three modular parts, one module each, implemented as written (Section 5.1):

  Part I   closures.py     position-closing propensity -> initial score x0
                           (closure = terminal contraction to <= c * running max, c = 0.005)
  Part II  propagation.py  x(k) = 1/2 (x0 + B x(k-1)), B = volume-weighted counterparty matrix,
                           iterate to relative l2 tolerance 1e-5 (Algorithm 1)
  Part III threshold.py    market-specific threshold theta* minimizing relative spillover Y(theta)
                           over [0.8, 0.99] with Y <= 0.1, slack zeta = 0.001 (Algorithm 2)

plus the two things the paper did not need but a single-asset batch-auction simulator does:

  pairing  pairing.py      who traded with whom inside a uniform-price clear: pro-rata split
  eval     evaluate.py     precision / recall / FPR against the ground truth (the paper had none)

Single-market adaptation (the one deliberate deviation, switchable): the paper's
x0_i = sum_m s_i^m 1{Q_i^m > 0} weights over MARKETS; with one asset that collapses to the binary
1{Q_i > 0}. ``n_windows=K`` treats K equal time windows as pseudo-markets (closures are still
found on the true, carried position path). ``n_windows=1`` is the literal formula.
"""
from .pairing import pair_trades, volume_graph, clear_stats
from .closures import find_closures, initial_scores
from .propagation import row_normalize, propagate, stationary
from .threshold import r_scores, spillover_curve, select_threshold
from .pipeline import DetectorConfig, Detection, detect, detect_run
from .evaluate import evaluate, evaluate_run, sweep

__all__ = [
    "pair_trades", "volume_graph", "clear_stats",
    "find_closures", "initial_scores",
    "row_normalize", "propagate", "stationary",
    "r_scores", "spillover_curve", "select_threshold",
    "DetectorConfig", "Detection", "detect", "detect_run",
    "evaluate", "evaluate_run", "sweep",
]
