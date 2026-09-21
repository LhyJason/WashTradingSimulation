r"""threshold.py -- Part III: market-specific threshold selection (Algorithm 2).

With one market the paper's definitions read:

  r_i      = min{ x_i, max_{j in N(i)} x_j }       largest theta at which wallet i would have a
                                                  flagged trade (N(i) = i's counterparties)
  Y(theta) = 1 - sum_{pairs} v_ij 1{min(r_i, r_j) >= theta}
                 / sum_{pairs} v_ij 1{max(r_i, r_j) >= theta}          relative spillover (eq. 4)
  Theta    = { theta in {r_i} : theta_lo <= theta <= theta_hi, Y(theta) <= Y_bar }
  theta*   = min argmin_{theta in Theta} max{zeta, Y(theta)}   if Theta is non-empty

Paper values: theta_lo = 0.8, theta_hi = 0.99, Y_bar = 0.1, zeta = 0.001. Trades between i and j
are then flagged iff min{x_i, x_j} >= theta*.

Empty Theta: the paper sets theta* = 1 "corresponding to no detected wash volume (though one
could equally well choose any theta > max r)". A perfectly closed simulated cluster can reach
x_i = 1 up to rounding, where "theta* = 1" would still flag it; so the empty case is represented
as theta* = +inf, which is the paper's stated meaning (nothing flagged).
"""
import numpy as np
import pandas as pd


def r_scores(x, V) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    neigh = np.where(V > 0, x[None, :], -np.inf).max(axis=1)
    return np.minimum(x, neigh)


def _pairs(V):
    iu, ju = np.triu_indices(V.shape[0], k=1)
    v = V[iu, ju]
    keep = v > 0
    return iu[keep], ju[keep], v[keep]


def spillover_curve(r, V, thetas) -> np.ndarray:
    """Y(theta) for each theta (NaN where no wallet has r >= theta)."""
    i, j, v = _pairs(V)
    rmin = np.minimum(r[i], r[j])
    rmax = np.maximum(r[i], r[j])
    out = np.full(len(thetas), np.nan)
    for k, th in enumerate(thetas):
        den = v[rmax >= th].sum()
        if den > 0:
            out[k] = 1.0 - v[rmin >= th].sum() / den
    return out


def select_threshold(x, V, theta_lo: float = 0.8, theta_hi: float = 0.99,
                     y_bar: float = 0.1, zeta: float = 0.001) -> dict:
    """Algorithm 2 for a single market. Returns theta_star (np.inf when Theta is empty) and the
    candidate table (theta, Y, objective, feasible)."""
    r = r_scores(x, V)
    cands = np.unique(r[(r >= theta_lo) & (r <= theta_hi)])
    Y = spillover_curve(r, V, cands)
    feasible = ~np.isnan(Y) & (Y <= y_bar)
    table = pd.DataFrame(dict(theta=cands, Y=Y, objective=np.maximum(zeta, Y), feasible=feasible))
    if feasible.any():
        obj = table.loc[table["feasible"], "objective"]
        theta_star = float(table.loc[obj.index[obj == obj.min()], "theta"].min())
    else:
        theta_star = np.inf
    return dict(theta_star=theta_star, table=table, r=r)
