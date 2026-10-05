r"""evaluate.py -- v2 evaluation: v1's metrics, plus same-group truth, per-group score check and
threshold curves.

The detector is v1's (``detection.v1.pipeline.detect``), unchanged. What v2 adds:

* **Same-group truth.** v1 calls a pair a wash trade iff both legs are wash orders. With several
  groups, a leg of group A can fill against a leg of group B (cross-group fill). v2 keeps v1's
  label (``is_wash``) and adds ``is_wash_same_group`` (both legs wash orders of the SAME group).
* **Per-group score check** (``group_table``). For a wash group G, with the detector's own
  counterparty volumes V (pro-rata pairing), let

      leak      l     = 1 - V[G, G].sum() / V[G, :].sum()       (share of G's volume with outsiders)
      outsider  s_bar = volume-weighted score of G's outside counterparties
      x0_bar, x_bar   = volume-weighted initial / final score of G's wallets

  If every wallet of G had the same score, the fixed point x = x0/2 + Bx/2 would give exactly

      x_bar = (x0_bar + l * s_bar) / (1 + l)                    (the leakage formula)

  so ``x_pred`` is the formula's value and ``x_bar - x_pred`` measures how far real groups are
  from that symmetric case. With s_bar ~ 0 and x0 = 1 the group falls below theta once
  l > (1 - theta) / theta (0.11 at theta = 0.9).
* **Threshold curves** (``threshold_curve``). Trade- and wallet-level counts at a grid of
  thresholds, so any threshold rule (e.g. one calibrated on control runs) can be evaluated later
  without re-running the detector.
"""
import numpy as np
import pandas as pd

from detection.v1.evaluate import _div, evaluate as _v1_evaluate, pair_metrics

THETA_GRID = np.round(np.arange(0.50, 1.0001, 0.01), 2)


def attach_truth(det, labels) -> tuple:
    """(pairs, wallets) with truth columns: pairs.is_wash, pairs.is_wash_same_group,
    wallets.is_wash, wallets.group_id."""
    legs = labels["trade_legs"].set_index("leg_id")
    is_w = legs["is_wash_leg"].astype(bool)
    grp = legs["group_id"]
    pairs = det.pairs.copy()
    bw = pairs["buy_leg_id"].map(is_w).to_numpy()
    sw = pairs["sell_leg_id"].map(is_w).to_numpy()
    pairs["is_wash"] = bw & sw
    pairs["is_wash_same_group"] = pairs["is_wash"].to_numpy() & (
        pairs["buy_leg_id"].map(grp).to_numpy() == pairs["sell_leg_id"].map(grp).to_numpy())
    wl = labels["wallets"].set_index("wallet")
    wallets = det.wallets.copy()
    wallets["is_wash"] = wl["is_wash"].astype(bool).reindex(wallets.index).fillna(False).to_numpy()
    wallets["group_id"] = wl["group_id"].reindex(wallets.index).fillna(-1).astype(int).to_numpy()
    return pairs, wallets


def evaluate(det, labels) -> dict:
    """v1's metric dict (same keys) + same-group trade metrics + pooled group leakage."""
    out = _v1_evaluate(det, labels)
    pairs, _ = attach_truth(det, labels)
    same = pairs.assign(is_wash=pairs["is_wash_same_group"])
    for tag, col in (("fixed", "flag_fixed"), ("star", "flag_star")):
        out.update({f"trade_{tag}_{k}_same": v for k, v in pair_metrics(same, col).items()
                    if k in ("precision", "recall")})
    out["cross_group_wash_volume"] = float(
        pairs.loc[pairs["is_wash"] & ~pairs["is_wash_same_group"], "qty"].sum())
    return out


def group_table(det, labels, theta=None) -> pd.DataFrame:
    """One row per wash group: size, leakage, outsider score, actual vs predicted mean score,
    and how many of its wallets the fixed-threshold rule flags."""
    theta = det.config.theta if theta is None else theta
    wl = labels["wallets"]
    wash = wl[wl["is_wash"].astype(bool)]
    pos = {w: i for i, w in enumerate(det.index)}
    x = det.wallets["x"].to_numpy()
    x0 = det.wallets["x0"].to_numpy()
    V = det.V
    rows = []
    for g, members in wash.groupby("group_id"):
        idx = np.array([pos[w] for w in members["wallet"] if w in pos], dtype=int)
        row = dict(group_id=int(g), n_wallets=int(len(members)), n_in_graph=int(len(idx)),
                   n_camouflage=int(members["camouflage"].astype(bool).sum()))
        if len(idx) == 0:
            rows.append(row)
            continue
        others = np.setdiff1d(np.arange(len(det.index)), idx)
        vol = V[idx].sum(axis=1)
        tot = vol.sum()
        inside = V[np.ix_(idx, idx)].sum()
        out_v = V[np.ix_(idx, others)]
        leak = 1.0 - inside / tot if tot else np.nan
        s_bar = float((out_v * x[others][None, :]).sum() / out_v.sum()) if out_v.sum() else np.nan
        x0_bar = float((vol * x0[idx]).sum() / tot) if tot else np.nan
        x_bar = float((vol * x[idx]).sum() / tot) if tot else np.nan
        x_pred = (x0_bar + leak * (s_bar if np.isfinite(s_bar) else 0.0)) / (1.0 + leak)
        flagged = det.wallets["flag_fixed"].to_numpy()[idx]
        row.update(volume=float(tot), leak=float(leak), s_bar=s_bar, x0_bar=x0_bar,
                   x_bar=x_bar, x_pred=float(x_pred), x_min=float(x[idx].min()),
                   x_max=float(x[idx].max()), n_flagged_fixed=int(flagged.sum()),
                   group_recall_fixed=_div(flagged.sum(), len(members)),
                   pred_detectable=bool(x_pred >= theta))
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_curve(det, labels, grid=THETA_GRID) -> pd.DataFrame:
    """Counts at each threshold: trade volume (true/false positives, both truth labels) and
    wallets (a wallet is flagged at theta iff r_i >= theta, as in v1)."""
    pairs, wallets = attach_truth(det, labels)
    p = pairs                       # all pairs, as in v1's metrics (self-pairs: score NaN, never flagged)
    s = p["score"].to_numpy()
    q = p["qty"].to_numpy()
    y = p["is_wash"].to_numpy()
    ys = p["is_wash_same_group"].to_numpy()
    r = wallets["r"].to_numpy()
    yw = wallets["is_wash"].to_numpy()
    rows = []
    for th in grid:
        f = s >= th
        fw = r >= th
        rows.append(dict(theta=float(th),
                         trade_tp=float(q[f & y].sum()), trade_fp=float(q[f & ~y].sum()),
                         trade_tp_same=float(q[f & ys].sum()),
                         wallet_tp=int((fw & yw).sum()), wallet_fp=int((fw & ~yw).sum())))
    df = pd.DataFrame(rows)
    df["trade_pos"] = float(q[y].sum())
    df["trade_neg"] = float(q[~y].sum())
    df["trade_pos_same"] = float(q[ys].sum())
    df["wallet_pos"] = int(yw.sum())
    df["wallet_neg"] = int((~yw).sum())
    return df
