r"""evaluate.py -- scoring a Detection against the ground truth (the part the paper could not do).

Ground truth, from labels/:
  * a PAIR (buy leg, sell leg) is a wash trade iff BOTH legs are wash orders (``is_wash_leg``);
    a camouflage wallet's ordinary ZI fill is not a wash leg, and an intercepting ZI's leg is
    not either, so interceptions count as honest volume;
  * a WALLET is a wash wallet iff it belongs to a wash group (``is_wash``).

Two levels, both reported at the fixed threshold (Algorithm 1) and at theta* (Algorithm 2):
  * trade level, VOLUME-weighted (the paper's own unit: "% of volume flagged");
  * wallet level, a wallet counts as flagged iff it has at least one flagged trade (the paper's
    "flags 14% of wallets").

Threshold-free curves: pairs ranked by min(x_i, x_j) (volume-weighted), wallets by r_i (a wallet
has a flagged trade at theta iff r_i >= theta). ROC-AUC and average precision summarize them.
Control runs have no positives: precision/recall are NaN there and FPR is the number to read.
"""
import numpy as np
import pandas as pd

from washtrade.v3.export import load_labels

from .pipeline import detect_run

_trapezoid = getattr(np, "trapezoid", None) or np.trapz     # numpy >= 2.0 renamed trapz


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def attach_truth(det, labels) -> tuple:
    """(pairs, wallets) with an ``is_wash`` column added."""
    legs = labels["trade_legs"].set_index("leg_id")["is_wash_leg"].astype(bool)
    pairs = det.pairs.copy()
    pairs["is_wash"] = pairs["buy_leg_id"].map(legs).to_numpy() & pairs["sell_leg_id"].map(legs).to_numpy()
    truth = labels["wallets"].set_index("wallet")["is_wash"].astype(bool)
    wallets = det.wallets.copy()
    wallets["is_wash"] = truth.reindex(wallets.index).fillna(False).to_numpy()
    return pairs, wallets


def pair_metrics(pairs, flag) -> dict:
    w = pairs["qty"].to_numpy()
    f = pairs[flag].to_numpy().astype(bool)
    y = pairs["is_wash"].to_numpy().astype(bool)
    tp, fp = w[f & y].sum(), w[f & ~y].sum()
    pos, neg = w[y].sum(), w[~y].sum()
    return dict(precision=_div(tp, tp + fp), recall=_div(tp, pos), fpr=_div(fp, neg),
                flagged_share=_div(tp + fp, w.sum()), true_share=_div(pos, w.sum()))


def wallet_metrics(wallets, flag) -> dict:
    f = wallets[flag].to_numpy().astype(bool)
    y = wallets["is_wash"].to_numpy().astype(bool)
    tp, fp = int((f & y).sum()), int((f & ~y).sum())
    return dict(precision=_div(tp, tp + fp), recall=_div(tp, y.sum()), fpr=_div(fp, (~y).sum()),
                n_flagged=int(f.sum()), n_true=int(y.sum()), n_wallets=int(len(y)))


def sweep(scores, y, w=None) -> pd.DataFrame:
    """Precision/recall/FPR at every distinct score threshold (flag iff score >= threshold).
    NaN scores are never flagged."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=bool)
    w = np.ones(len(s)) if w is None else np.asarray(w, dtype=float)
    s = np.where(np.isnan(s), -np.inf, s)
    order = np.argsort(-s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    # index of the last element of each run of equal scores (s[1:] != s[:-1], not np.diff: -inf - -inf is NaN)
    last = np.r_[np.nonzero(s[1:] != s[:-1])[0], len(s) - 1] if len(s) else np.array([], dtype=int)
    tp = np.cumsum(w * y)[last] if len(s) else np.array([])
    fp = np.cumsum(w * ~y)[last] if len(s) else np.array([])
    P, N = w[y].sum(), w[~y].sum()
    df = pd.DataFrame(dict(threshold=s[last] if len(s) else [], tp=tp, fp=fp))
    df = df[np.isfinite(df["threshold"])].reset_index(drop=True)
    df["precision"] = df["tp"] / (df["tp"] + df["fp"])
    df["recall"] = df["tp"] / P if P > 0 else np.nan
    df["fpr"] = df["fp"] / N if N > 0 else np.nan
    return df


def curve_summary(curve: pd.DataFrame) -> dict:
    if curve.empty or curve["recall"].isna().all() or curve["fpr"].isna().all():
        return dict(roc_auc=float("nan"), avg_precision=float("nan"))
    fpr = np.r_[0.0, curve["fpr"].to_numpy(), 1.0]
    tpr = np.r_[0.0, curve["recall"].to_numpy(), 1.0]
    rec = np.r_[0.0, curve["recall"].to_numpy()]
    ap = float(np.sum(np.diff(rec) * curve["precision"].to_numpy()))
    return dict(roc_auc=float(_trapezoid(tpr, fpr)), avg_precision=ap)


def evaluate(det, labels) -> dict:
    pairs, wallets = attach_truth(det, labels)
    out = dict(theta_star=det.theta_star, alg2_detects=bool(np.isfinite(det.theta_star)),
               n_iter=det.n_iter, **{f"graph_{k}": v for k, v in det.stats.items()})
    for tag, col in (("fixed", "flag_fixed"), ("star", "flag_star")):
        out.update({f"trade_{tag}_{k}": v for k, v in pair_metrics(pairs, col).items()})
        out.update({f"wallet_{tag}_{k}": v for k, v in wallet_metrics(wallets, col).items()})
    pc = sweep(pairs["score"], pairs["is_wash"], pairs["qty"])
    wc = sweep(wallets["r"], wallets["is_wash"])
    out.update({f"trade_{k}": v for k, v in curve_summary(pc).items()})
    out.update({f"wallet_{k}": v for k, v in curve_summary(wc).items()})
    # how well the raw scores separate the classes
    out["x_wash_mean"] = float(wallets.loc[wallets["is_wash"], "x"].mean()) if wallets["is_wash"].any() else float("nan")
    out["x_honest_mean"] = float(wallets.loc[~wallets["is_wash"], "x"].mean()) if (~wallets["is_wash"]).any() else float("nan")
    out["x_honest_max"] = float(wallets.loc[~wallets["is_wash"], "x"].max()) if (~wallets["is_wash"]).any() else float("nan")
    return out


def evaluate_run(run_dir, det=None, config=None) -> dict:
    det = det if det is not None else detect_run(run_dir, config)
    return evaluate(det, load_labels(run_dir))
