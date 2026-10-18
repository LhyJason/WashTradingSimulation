r"""fusion.py -- from channel scores to one decision, calibrated on control runs.

Channels live on different scales: a round-trip score of 0.6 is ordinary in an active market and
damning in a thin one. Each channel score is therefore put on a common scale first, using the
background's control runs (no wash group, the same seeds as every sweep):

    u_k(s) = share of honest pair volume in the control runs whose channel-k score is >= s

u_k is the channel's volume-weighted tail probability ("how often does honest trading look at
least this suspicious?").

**Score floor.** A pair's channel-k score only counts as evidence once it reaches a floor f:

    u_k^f(s) = u_k(s) if s >= f, else 1

The floor is needed because a channel can be exactly 0 on almost all honest volume. The repeat
channels are an example. Their control tail then ends at tiny positive scores, so the calibrated
threshold sits just above 0. In a wash run, the network step gives every wallet connected to the
wash group a small positive score, and such a threshold flags it. In the scenarios sweep, without
a floor, up to 95% of honest volume was flagged in wash runs while the control FPR was 1%. Scores
are network-smoothed shares of volume, so f = 0.5 reads "at least half of both wallets' volume
looks wash-like". This is the same kind of absolute level as Sethi's theta = 0.9 (MAIN_FLOOR).
Floors 0, 0.3 and 0.7 are kept as a sensitivity check. The floor does not affect the closure
channel, whose calibrated thresholds are all above 0.8.

The fused score of a pair over a channel set S is

    z_S = 1 - min_{k in S} u_k^f(score_k)

so a pair is as suspicious as its most extreme channel. A single channel's method score is its
raw score where it reaches the floor, and -1 below it (never flagged).

Threshold (as in detection v3): for every background, method and floor, ``theta`` is the smallest
threshold whose POOLED control FPR (false-positive volume / honest volume) is at most the target.
It is found exactly from the control scores, not on a 0.01 grid as in v3. If even the highest
control score breaks the target, the threshold sits just above it (+1e-9): nothing in the control
runs is flagged, but a wash pair can still be. Control runs are scored with the same u_k as
everything else, so the pooled control FPR at theta is <= target by construction. FPR in WASH
runs is not controlled. Honest wallets that trade with wash wallets pick up their scores through
the network step (spillover), so the FPR in wash runs is reported separately.

Ranking quality (AP) is threshold-free and uses the floor-0 scores (raw score, or z without floor).
"""
import numpy as np
import pandas as pd

from .detector import CHANNELS, NOPROP

TARGETS = {"1": 0.01, "0.1": 0.001, "5": 0.05}     # tag -> pooled control FPR; "1" is the main one
MAIN_TARGET = "1"
FLOORS = (0.0, 0.3, 0.5, 0.7)
MAIN_FLOOR = 0.5
ABOVE = 1e-9            # "just above the highest control score" (kept clear of float noise)
BELOW = -1.0            # single-channel method score below the floor
ROUNDTRIP = [c for c in CHANNELS if c.startswith("roundtrip_")]
REPEAT = [c for c in CHANNELS if c.startswith("repeat_")]
FUSIONS = {
    "closure+roundtrip": ["closure"] + ROUNDTRIP,
    "closure+repeat":    ["closure"] + REPEAT,
    "v4":                list(CHANNELS),
}
SINGLE = list(CHANNELS) + [f"{c}_noprop" for c in NOPROP]
METHODS = SINGLE + list(FUSIONS)


def floor_tag(f) -> str:
    return f"f{f:g}"


class Survival:
    """Volume-weighted tail probability of one score over the honest control volume."""

    def __init__(self, scores, qty, total):
        s = np.asarray(scores, dtype=float)
        q = np.asarray(qty, dtype=float)
        keep = ~np.isnan(s)
        s, q = s[keep], q[keep]
        o = np.argsort(s, kind="mergesort")
        self.s = s[o]
        # tail[k] = volume share of control scores >= s[k]; one extra 0 for "above them all"
        self.tail = np.r_[np.cumsum(q[o][::-1])[::-1] / float(total), 0.0]

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        k = np.searchsorted(self.s, np.where(np.isnan(x), 0.0, x), side="left")
        return np.where(np.isnan(x), np.nan, self.tail[k])


class Null:
    """Per background: the survival function of every single score, built from control pairs.

    ``control``: one row per control pair with columns background, qty and one per score in
    ``SINGLE`` (NaN for self-pairs). The honest volume of a background is all its control pair
    volume, self-pairs included (they are never flagged), as in v1-v3's FPR."""

    def __init__(self, control: pd.DataFrame):
        self.maps = {}
        self.total = {}
        for bg, d in control.groupby("background"):
            tot = float(d["qty"].sum())
            self.total[bg] = tot
            self.maps[bg] = {c: Survival(d[c], d["qty"], tot) for c in SINGLE}

    @property
    def backgrounds(self):
        return sorted(self.maps)

    def fused(self, bg, pairs: pd.DataFrame, channels, floor=MAIN_FLOOR) -> np.ndarray:
        """z_S for every row of ``pairs`` (NaN for self-pairs)."""
        us = []
        for c in channels:
            s = pairs[c].to_numpy(dtype=float)
            u = self.maps[bg][c](s)
            us.append(np.where(s >= floor, u, np.where(np.isnan(s), np.nan, 1.0)))
        u = np.vstack(us)
        with np.errstate(invalid="ignore"):
            z = 1.0 - np.min(u, axis=0)
        return np.where(np.isnan(u).any(axis=0), np.nan, z)

    def method_scores(self, bg, pairs: pd.DataFrame, floor=MAIN_FLOOR) -> pd.DataFrame:
        """One column per method: the raw score (BELOW under the floor) for single channels,
        z_S for fusions."""
        out = {}
        for m in SINGLE:
            s = pairs[m].to_numpy(dtype=float)
            out[m] = np.where(s >= floor, s, np.where(np.isnan(s), np.nan, BELOW))
        for name, chans in FUSIONS.items():
            out[name] = self.fused(bg, pairs, chans, floor)
        return pd.DataFrame(out, index=pairs.index)


def theta_for(scores, qty, total, target) -> float:
    """Smallest threshold with sum(qty[score >= theta]) / total <= target; if none, just above
    the highest score."""
    s = np.asarray(scores, dtype=float)
    q = np.asarray(qty, dtype=float)
    keep = ~np.isnan(s)
    s, q = s[keep], q[keep]
    if len(s) == 0:
        return -np.inf
    o = np.argsort(-s, kind="mergesort")
    s, q = s[o], q[o]
    last = np.r_[np.nonzero(s[1:] != s[:-1])[0], len(s) - 1]      # end of each run of ties
    fpr = np.cumsum(q)[last] / float(total)
    ok = np.nonzero(fpr <= target)[0]
    if len(ok) == 0:
        return float(s[0] + ABOVE)
    return float(s[last[ok.max()]])


def thresholds(control: pd.DataFrame, null: Null, targets=TARGETS, floors=FLOORS) -> pd.DataFrame:
    """One row per (background, method, floor): theta for each target and the pooled control FPR
    there. At floors other than MAIN_FLOOR only the main target is computed."""
    rows = []
    for bg, d in control.groupby("background"):
        tot = null.total[bg]
        q = d["qty"].to_numpy()
        for fl in floors:
            sc = null.method_scores(bg, d, fl)
            tg = targets if fl == MAIN_FLOOR else {MAIN_TARGET: targets[MAIN_TARGET]}
            for m in METHODS:
                row = dict(background=bg, method=m, floor=fl,
                           n_control_runs=int(d["run_id"].nunique()), honest_volume=tot)
                for tag, tgt in tg.items():
                    th = theta_for(sc[m], q, tot, tgt)
                    row[f"theta_{tag}"] = th
                    row[f"fpr_{tag}"] = float(q[sc[m].to_numpy() >= th].sum() / tot)
                rows.append(row)
    return pd.DataFrame(rows)
