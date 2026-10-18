r"""evaluate.py -- one run's v4 detection scored against the labels, for every method.

Truth is v2's: a pair is wash iff both legs are wash legs (``is_wash``), and same-group wash iff
both legs belong to the same group (``is_wash_same_group``). A wallet is a wash wallet iff it is in
a wash group. As in v1, a wallet counts as flagged iff it has at least one flagged trade, and
wallet metrics cover the wallets in the graph only.

Per (run, method):
  * at the main floor (0.5) and the background's calibrated threshold for a pooled control FPR of
    1%: volume counts (trade_tp, trade_fp, trade_tp_same, ...), the ratios built from them, and
    wallet counts and ratios;
  * at the main floor, trade counts for the other targets (``trade_tp_0.1`` etc.: 0.1%, 5%);
  * at the other floors, trade counts for the 1% target (``trade_tp_f0``, ``trade_tp_f0.3``, ...);
  * threshold-free, on the floor-0 scores: volume-weighted average precision and ROC-AUC (v1's
    ``sweep`` / ``curve_summary``, ties handled).
Counts are kept so that results can be pooled over runs.
"""
import numpy as np
import pandas as pd

from detection.v1.evaluate import _div, curve_summary, sweep
from detection.v2.evaluate import attach_truth

from .fusion import FLOORS, MAIN_FLOOR, MAIN_TARGET, METHODS, TARGETS, floor_tag


def score_run(det, labels, null, thresholds: pd.DataFrame, background) -> list:
    """One dict per method. ``thresholds``: rows of this background (method, floor, theta_<tag>)."""
    pairs, wallets = attach_truth(det, labels)
    th = thresholds.set_index(["method", "floor"])
    scores = {fl: null.method_scores(background, pairs, fl) for fl in FLOORS}
    q = pairs["qty"].to_numpy().astype(float)
    y = pairs["is_wash"].to_numpy().astype(bool)
    ys = pairs["is_wash_same_group"].to_numpy().astype(bool)
    pos, neg, pos_same = q[y].sum(), q[~y].sum(), q[ys].sum()
    buyer = pairs["buyer"].to_numpy()
    seller = pairs["seller"].to_numpy()
    yw = wallets["is_wash"].to_numpy().astype(bool)

    def flags(m, fl, tag):
        s = scores[fl][m].to_numpy()
        return np.where(np.isnan(s), -np.inf, s) >= th.loc[(m, fl), f"theta_{tag}"]

    rows = []
    for m in METHODS:
        row = dict(method=m, trade_pos=pos, trade_neg=neg, trade_pos_same=pos_same,
                   wallet_pos=int(yw.sum()), wallet_neg=int((~yw).sum()))
        f = flags(m, MAIN_FLOOR, MAIN_TARGET)
        row["trade_tp"] = float(q[f & y].sum())
        row["trade_fp"] = float(q[f & ~y].sum())
        row["trade_tp_same"] = float(q[f & ys].sum())
        fw = wallets.index.isin(set(buyer[f]) | set(seller[f]))
        row["wallet_tp"] = int((fw & yw).sum())
        row["wallet_fp"] = int((fw & ~yw).sum())
        for tag in TARGETS:
            if tag != MAIN_TARGET:
                f = flags(m, MAIN_FLOOR, tag)
                row[f"trade_tp_{tag}"] = float(q[f & y].sum())
                row[f"trade_fp_{tag}"] = float(q[f & ~y].sum())
        for fl in FLOORS:
            if fl != MAIN_FLOOR:
                f = flags(m, fl, MAIN_TARGET)
                row[f"trade_tp_{floor_tag(fl)}"] = float(q[f & y].sum())
                row[f"trade_fp_{floor_tag(fl)}"] = float(q[f & ~y].sum())
        row.update(trade_recall=_div(row["trade_tp"], pos),
                   trade_precision=_div(row["trade_tp"], row["trade_tp"] + row["trade_fp"]),
                   trade_fpr=_div(row["trade_fp"], neg),
                   trade_recall_same=_div(row["trade_tp_same"], pos_same),
                   wallet_recall=_div(row["wallet_tp"], row["wallet_pos"]),
                   wallet_fpr=_div(row["wallet_fp"], row["wallet_neg"]))
        cs = curve_summary(sweep(scores[0.0][m].to_numpy(), y, q))
        row.update(trade_ap=cs["avg_precision"], trade_roc_auc=cs["roc_auc"])
        rows.append(row)
    return rows


def wallet_table(det, labels) -> pd.DataFrame:
    """One row per wallet in the graph: every channel's x0 and x, volume, degree and the truth."""
    _, w = attach_truth(det, labels)
    wl = labels["wallets"].set_index("wallet")
    w["camouflage"] = wl["camouflage"].astype(bool).reindex(w.index).fillna(False).to_numpy()
    return w.reset_index()
