r"""detector.py -- detection v6 on one tape: v4's channels plus the cluster channel.

    trades --(v4: pairing, graph, seven channels, propagation)--> pair scores
           --(cluster.py: Louvain on V, wallet excess, pair min)--> cluster, cluster_r2

``detect_v6`` reads only the observable tape. The v4 columns are exactly ``detect_v4``'s.
"""
from dataclasses import dataclass, field

import numpy as np

from detection.v4.detector import V4Config, V4Detection, detect_v4

from .cluster import RESOLUTIONS, SEED, channel_name, communities, modularity, pair_scores, wallet_scores


@dataclass
class V6Config:
    v4: V4Config = field(default_factory=V4Config)
    resolutions: tuple = RESOLUTIONS
    seed: int = SEED


@dataclass
class V6Detection(V4Detection):
    graph_stats: dict = field(default_factory=dict)     # per resolution: n_comm, modularity


def cluster_channels(config: V6Config = None):
    cfg = config or V6Config()
    return [channel_name(r) for r in cfg.resolutions]


def detect_v6(trades, config: V6Config = None) -> V6Detection:
    cfg = config or V6Config()
    d4 = detect_v4(trades, cfg.v4)
    pairs, wallets = d4.pairs, d4.wallets
    pos = {w: i for i, w in enumerate(d4.index)}
    ok = pairs["in_graph"].to_numpy()
    bi = pairs["buyer"].map(pos).fillna(-1).to_numpy()
    si = pairs["seller"].map(pos).fillna(-1).to_numpy()
    stats = {}
    for r in cfg.resolutions:
        ch = channel_name(r)
        lab = communities(d4.V, r, cfg.seed)
        s = wallet_scores(d4.V, lab)
        wallets[f"comm_{ch}"] = lab
        wallets[f"x_{ch}"] = s
        pairs[ch] = pair_scores(lab, s, bi, si, ok)
        stats[ch] = dict(n_comm=int(len(np.unique(lab))) if len(lab) else 0,
                         modularity=modularity(d4.V, lab) if len(lab) else float("nan"))
    return V6Detection(config=d4.config, pairs=pairs, wallets=wallets, index=d4.index, V=d4.V,
                       n_iter=d4.n_iter, graph_stats=stats)
