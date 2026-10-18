r"""channels.py -- Part I evidence channels: what makes a wallet's trading look like wash trading.

Every channel gives each wallet an initial score x0 in [0, 1], a share of the wallet's volume, as
Sethi's closing score does. Each then goes through the same network step (Part II).

  closure       Sethi et al., unchanged (detection v1): x0 = 1{the wallet ever closes its position}.
  roundtrip_W   FAST ROUND TRIPS. The wallet's unit trades, in time order, are matched first-in
                first-out with an opposite unit of the same wallet traded at most W steps later
                (a buy undone by a sell, or a sell by a buy). x0 = share of its units that are matched.
                An honest trader also undoes positions, but only over the long times between its
                arrivals; a wash round undoes them within the round.
  repeat_W      SAME-PAIR REPEATS. A trade between seller s and buyer b is a repeat if the same
                ordered pair (s -> b) trades again within W steps, before or after it. x0 = share of
                the wallet's volume in repeat trades. Honest traders meet a given counterparty, in a
                given direction, twice within a short window only by chance; a wash round that
                trades Q units one at a time puts Q consecutive trades on each of its edges.

The closure channel ignores time, which is why one closure anywhere in the run is enough and why a
rotating pool defeats it (detection v3). The two new channels look only at short windows, so they
do not care how many rounds a wallet takes part in.

Conventions (as in v1): all legs of one wallet in one clear are netted into one position step
(``per_clear_sequence``); self-pairs (one wallet on both sides of a clear) are ignored.
"""
import numpy as np
import pandas as pd

from detection.v1.closures import initial_scores, per_clear_sequence

ROUNDTRIP_WINDOWS = (50, 100, 200)
REPEAT_WINDOWS = (20, 50, 200)


def roundtrip_share(times, signed, window) -> float:
    """Share of a wallet's units matched FIFO with an opposite unit at most ``window`` steps later.

    ``times`` / ``signed``: the wallet's position steps in time order (nonzero net quantity per
    clear; + buy, - sell). A step of size k is k units at the same time."""
    t = np.repeat(np.asarray(times, dtype=float), np.abs(np.asarray(signed, dtype=int)))
    s = np.repeat(np.sign(np.asarray(signed, dtype=int)), np.abs(np.asarray(signed, dtype=int)))
    if len(t) == 0:
        return 0.0
    open_t, open_s = [], []          # FIFO of unmatched units: all have the same sign
    matched = 0
    for ti, si in zip(t, s):
        while open_t and ti - open_t[0] > window:      # too old to be undone any more
            open_t.pop(0)
            open_s.pop(0)
        if open_t and open_s[0] == -si:
            open_t.pop(0)
            open_s.pop(0)
            matched += 2
        else:
            open_t.append(ti)
            open_s.append(si)
    return matched / len(t)


def roundtrip_scores(trades: pd.DataFrame, windows=ROUNDTRIP_WINDOWS) -> pd.DataFrame:
    """Index wallet; one column per window (``roundtrip_W``)."""
    cols = [f"roundtrip_{w}" for w in windows]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    seq = per_clear_sequence(trades)
    seq = seq[seq["signed"] != 0]
    rows = {}
    for wallet, g in seq.groupby("wallet", sort=True):
        tm, sg = g["time"].to_numpy(), g["signed"].to_numpy()
        rows[wallet] = [roundtrip_share(tm, sg, w) for w in windows]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    # wallets whose every clear nets to zero have no position steps: nothing to undo
    allw = trades["wallet"].unique()
    return out.reindex(sorted(allw)).fillna(0.0)


def repeat_flags(pairs: pd.DataFrame, window) -> np.ndarray:
    """Boolean per pair row: the same ordered (seller, buyer) pair trades again within ``window``
    steps (before or after). Self-pairs are never repeats."""
    flag = np.zeros(len(pairs), dtype=bool)
    if pairs.empty:
        return flag
    ok = (pairs["seller"] != pairs["buyer"]).to_numpy()
    p = pairs.loc[ok, ["seller", "buyer", "time"]]
    order = np.lexsort((p["time"].to_numpy(), p["buyer"].to_numpy(), p["seller"].to_numpy()))
    s = p["seller"].to_numpy()[order]
    b = p["buyer"].to_numpy()[order]
    t = p["time"].to_numpy()[order].astype(float)
    same = (s[1:] == s[:-1]) & (b[1:] == b[:-1])
    close = same & (t[1:] - t[:-1] <= window)
    f = np.zeros(len(t), dtype=bool)
    f[1:] |= close
    f[:-1] |= close
    rows = np.nonzero(ok)[0][order]
    flag[rows] = f
    return flag


def repeat_scores(pairs: pd.DataFrame, windows=REPEAT_WINDOWS) -> pd.DataFrame:
    """Index wallet; one column per window (``repeat_W``): share of the wallet's (non-self) pair
    volume in repeat trades."""
    cols = [f"repeat_{w}" for w in windows]
    ok = (pairs["seller"] != pairs["buyer"]).to_numpy() if len(pairs) else np.zeros(0, bool)
    if not ok.any():
        return pd.DataFrame(columns=cols)
    who = np.concatenate([pairs["seller"].to_numpy()[ok], pairs["buyer"].to_numpy()[ok]])
    q = pairs["qty"].to_numpy()[ok].astype(float)
    vol = pd.Series(np.concatenate([q, q]), index=who).groupby(level=0).sum()
    out = {}
    for w in windows:
        f = repeat_flags(pairs, w)[ok]
        rep = pd.Series(np.concatenate([q * f, q * f]), index=who).groupby(level=0).sum()
        out[f"repeat_{w}"] = (rep / vol).clip(0.0, 1.0)
    return pd.DataFrame(out, columns=cols)


def channel_scores(trades: pd.DataFrame, pairs: pd.DataFrame, index, c: float = 0.005,
                   roundtrip_windows=ROUNDTRIP_WINDOWS,
                   repeat_windows=REPEAT_WINDOWS) -> pd.DataFrame:
    """Every channel's x0 for the wallets in ``index`` (the volume graph's nodes)."""
    clo = initial_scores(trades, c=c)["x0"].rename("closure")
    rt = roundtrip_scores(trades, roundtrip_windows)
    rp = repeat_scores(pairs, repeat_windows)
    out = pd.concat([clo, rt, rp], axis=1).reindex(index)
    return out.fillna(0.0).astype(float)
