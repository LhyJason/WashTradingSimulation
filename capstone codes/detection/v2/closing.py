r"""closing.py -- why honest wallets "close": the random-walk benchmark, and a per-wallet table.

The market has no settlement or forced liquidation. A "closure" in Sirolly et al. is a pattern on
the tape: a wallet's net position comes back to (almost) zero, X_tau <= c * M with c = 0.005. Here
positions are integers no larger than q_max = 10, so c * M < 1 and a closure means the position
returns exactly to 0.

A ZI picks buy or sell with a fair coin on every arrival (HBL too, through the adapter), so its
position path is close to a simple random walk. For a simple random walk the probability of NOT
returning to 0 within n steps is

    u(n) = C(2m, m) / 4^m,   m = floor(n / 2)

so the probability of at least one closure after n position-changing trades is 1 - u(n)
(0.50 at n = 2, 0.69 at n = 6, 0.82 at n = 20, 0.87 at n = 40). ``wallet_table`` puts this
prediction next to each wallet's observed closures; comparing the two shows when honest wallets
behave like random walks (active markets) and when they do not (the r = 0.01 market, where honest
trades are few and driven by private values).
"""
from math import comb

import numpy as np
import pandas as pd

from detection.v1.closures import per_clear_sequence


def p_no_return(n: int) -> float:
    """P(simple random walk started at 0 does not revisit 0 within n steps)."""
    m = int(n) // 2
    return comb(2 * m, m) / 4 ** m


def p_return(n: int) -> float:
    """P(at least one return to 0 within n steps) = 1 - p_no_return(n)."""
    return 1.0 - p_no_return(n)


def wallet_table(det, trades: pd.DataFrame, labels) -> pd.DataFrame:
    """One row per wallet in the detector's graph: position steps, closures, the random-walk
    prediction, scores, flags and the ground truth (is_wash, group_id, camouflage)."""
    seq = per_clear_sequence(trades)
    steps = seq[seq["signed"] != 0].groupby("wallet").size()
    w = det.wallets.copy()
    w["n_steps"] = steps.reindex(w.index).fillna(0).astype(int).to_numpy()
    w["closed"] = w["n_closures"].fillna(0).to_numpy() > 0
    w["rw_p_close"] = [p_return(n) for n in w["n_steps"]]
    wl = labels["wallets"].set_index("wallet")
    w["is_wash"] = wl["is_wash"].astype(bool).reindex(w.index).fillna(False).to_numpy()
    w["group_id"] = wl["group_id"].reindex(w.index).fillna(-1).astype(int).to_numpy()
    w["camouflage"] = wl["camouflage"].astype(bool).reindex(w.index).fillna(False).to_numpy()
    cols = ["is_wash", "group_id", "camouflage", "n_steps", "n_closures", "closed", "rw_p_close",
            "volume", "degree", "x0", "x", "r", "flag_fixed", "flag_star"]
    return w[cols].reset_index()
