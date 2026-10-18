r"""detector.py -- the v4 detector on one tape: every channel's wallet scores and pair scores.

    trades --pairing (v1)--> pairs --volume_graph (v1)--> V, B = row-normalized V
           --channels--> x0[channel] --propagation (v1, rho = 1/2)--> x[channel]
    pair score[channel] = min(x_buyer, x_seller)            (as in Algorithm 1)

The ``closure`` channel is Sethi et al.'s detector exactly as in v1 (``detection.v1.pipeline
.detect`` with its default config): same x0, same propagation, same pair scores.

Two ablations skip the network step for one channel each (``<channel>_noprop``: x = x0). They
show how much of a channel's power comes from Part II.

``detect_v4`` reads only the observable tape. Turning channel scores into one decision needs the
control runs, see ``fusion.py``.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from detection.v1.pairing import pair_trades, volume_graph
from detection.v1.propagation import propagate, row_normalize

from .channels import REPEAT_WINDOWS, ROUNDTRIP_WINDOWS, channel_scores

NOPROP = ("roundtrip_100", "repeat_50")


def channel_names(roundtrip_windows=ROUNDTRIP_WINDOWS, repeat_windows=REPEAT_WINDOWS):
    return (["closure"] + [f"roundtrip_{w}" for w in roundtrip_windows]
            + [f"repeat_{w}" for w in repeat_windows])


CHANNELS = channel_names()
SCORES = CHANNELS + [f"{c}_noprop" for c in NOPROP]


@dataclass
class V4Config:
    c: float = 0.005
    rho: float = 0.5
    tol: float = 1e-5
    roundtrip_windows: tuple = ROUNDTRIP_WINDOWS
    repeat_windows: tuple = REPEAT_WINDOWS
    noprop: tuple = NOPROP


@dataclass
class V4Detection:
    config: V4Config
    pairs: pd.DataFrame          # pair table + in_graph + one score column per channel / ablation
    wallets: pd.DataFrame        # index wallet: x0_<channel>, x_<channel>, volume, degree
    index: list
    V: np.ndarray
    n_iter: dict = field(default_factory=dict)

    @property
    def score_names(self):
        cfg = self.config
        return (channel_names(cfg.roundtrip_windows, cfg.repeat_windows)
                + [f"{c}_noprop" for c in cfg.noprop])


def detect_v4(trades: pd.DataFrame, config: V4Config = None) -> V4Detection:
    cfg = config or V4Config()
    pairs = pair_trades(trades)
    index, V, _ = volume_graph(pairs)
    n = len(index)
    B = row_normalize(V) if n else V
    x0 = channel_scores(trades, pairs, index, c=cfg.c, roundtrip_windows=cfg.roundtrip_windows,
                        repeat_windows=cfg.repeat_windows)
    names = channel_names(cfg.roundtrip_windows, cfg.repeat_windows)

    pos = {w: i for i, w in enumerate(index)}
    pairs = pairs.copy()
    pairs["in_graph"] = (pairs["buyer"] != pairs["seller"]).to_numpy()
    ok = pairs["in_graph"].to_numpy()
    bi = pairs["buyer"].map(pos).to_numpy()
    si = pairs["seller"].map(pos).to_numpy()

    wallets = pd.DataFrame(index=pd.Index(index, name="wallet"))
    n_iter = {}

    def pair_score(x):
        out = np.full(len(pairs), np.nan)
        if ok.any():
            out[ok] = np.minimum(x[bi[ok].astype(int)], x[si[ok].astype(int)])
        return out

    for ch in names:
        v0 = x0[ch].to_numpy() if n else np.zeros(0)
        if n and v0.any():
            x, _, k = propagate(v0, B, rho=cfg.rho, tol=cfg.tol)
        else:                                   # all-zero x0 stays zero (and avoids 0/0)
            x, k = v0.copy(), 0
        wallets[f"x0_{ch}"] = v0
        wallets[f"x_{ch}"] = x
        pairs[ch] = pair_score(x)
        n_iter[ch] = k
        if ch in cfg.noprop:
            pairs[f"{ch}_noprop"] = pair_score(v0)
    wallets["volume"] = V.sum(axis=1)
    wallets["degree"] = (V > 0).sum(axis=1)
    return V4Detection(config=cfg, pairs=pairs, wallets=wallets, index=index, V=V, n_iter=n_iter)
