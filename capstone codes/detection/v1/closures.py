r"""closures.py -- Part I: position-closing propensity (Sirolly et al. 2026, Section 5.1).

Definitions, verbatim from the paper, for one wallet's chronological trades:

  * a trade that flips the sign of the position is split in two: one leg to zero, one opening
    the opposite side;
  * P_t = net position after trade t, X_t = |P_t|; trade t is an EXPANSION if X_t > X_{t-1} and a
    CONTRACTION if X_t < X_{t-1};
  * a TERMINAL contraction is a contraction immediately followed by an expansion, or the
    wallet's last trade;
  * a terminal contraction tau_l is a CLOSURE if  X_{tau_l} <= c * M_l,
    M_l = max{ X_t : tau_{l-1} < t < tau_l }  (max position since the previous closure);
    closures are found iteratively; c = 0.005.

Initial score:  x0_i = sum_m s_i^m * 1{Q_i^m > 0},  s_i^m = share of i's volume in market m.

Single-asset adaptation. The simulator has ONE market, where that formula is just 1{Q_i > 0}.
``n_windows = K`` splits the horizon into K equal windows and uses them as pseudo-markets: a
closure counts for the window its trade falls in, volume shares are per window. Positions are
NOT reset at window boundaries -- closures are always found on the true position path, so the
windows only change the volume weighting. ``n_windows = 1`` reproduces the literal formula.

Batch clearing detail: all legs of one wallet in one clear are simultaneous, so they are netted
into one trade (a wallet buying 1 and selling 1 in the same clear does not move its position);
gross quantity still counts toward its volume.
"""
import numpy as np
import pandas as pd

SELL = "SELL"


def find_closures(signed, c: float = 0.005) -> list:
    """Indices (into ``signed``) of the trades that are closures. ``signed`` = chronological
    signed quantities (+ buy, - sell), zero entries not allowed."""
    X, src = [], []
    P = 0.0
    for k, q in enumerate(signed):
        new = P + q
        if P != 0 and new != 0 and (P > 0) != (new > 0):     # reversal -> split in two
            X.append(0.0)
            src.append(k)
        X.append(abs(new))
        src.append(k)
        P = new

    closures = []
    M = 0.0                  # max X since the previous closure (excluding the current trade)
    prev = 0.0
    n = len(X)
    for t in range(n):
        x = X[t]
        terminal = x < prev and (t == n - 1 or X[t + 1] > x)
        if terminal and M > 0 and x <= c * M:
            closures.append(src[t])
            M = 0.0
        else:
            M = max(M, x)
        prev = x
    return closures


def per_clear_sequence(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per (wallet, clear): net signed quantity and gross quantity."""
    signed = np.where(trades["side"] == SELL, -trades["qty"], trades["qty"])
    df = pd.DataFrame(dict(wallet=trades["wallet"], time=trades["time"],
                           signed=signed, gross=trades["qty"]))
    return df.groupby(["wallet", "time"], sort=True)[["signed", "gross"]].sum().reset_index()


def initial_scores(trades: pd.DataFrame, c: float = 0.005, n_windows: int = 1,
                   horizon=None) -> pd.DataFrame:
    """Part I for every wallet on the tape. Index = wallet; columns x0, n_closures, volume,
    n_clears, n_closed_windows."""
    cols = ["x0", "n_closures", "volume", "n_clears", "n_closed_windows"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    K = max(1, int(n_windows))
    H = float(horizon) if horizon is not None else float(trades["time"].max() + 1)

    def window(t):
        return np.minimum(K - 1, np.floor(np.asarray(t, dtype=float) * K / H).astype(int))

    rows = {}
    for wallet, g in per_clear_sequence(trades).groupby("wallet", sort=True):
        nz = g[g["signed"] != 0]
        idx = find_closures(nz["signed"].tolist(), c)
        closed_w = set(window(nz["time"].to_numpy()[idx]).tolist()) if idx else set()
        vol_w = g.groupby(window(g["time"].to_numpy()))["gross"].sum()
        total = float(vol_w.sum())
        x0 = float(vol_w[vol_w.index.isin(closed_w)].sum() / total) if total else 0.0
        rows[wallet] = dict(x0=x0, n_closures=len(idx), volume=total,
                            n_clears=int(len(g)), n_closed_windows=len(closed_w))
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)
