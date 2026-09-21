r"""pipeline.py -- the full detector: observable tape -> wallet scores -> flagged trades.

    trades.csv --pairing--> pairs --volume_graph--> V, B
               --closures--> x0 --propagation--> x --threshold--> theta*
    flag pair (i, j) iff min(x_i, x_j) >= theta   (Algorithm 1: fixed theta; Algorithm 2: theta*)

``detect_run`` reads ONLY ``observable/trades.csv`` plus the market horizon (``sim_time``, public
information) from meta.json. Nothing under labels/ is opened here.
"""
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from washtrade.v3.export import load_meta, load_observable

from .closures import initial_scores
from .pairing import clear_stats, pair_trades, volume_graph
from .propagation import propagate, row_normalize
from .threshold import select_threshold


@dataclass
class DetectorConfig:
    # Part I
    c: float = 0.005
    n_windows: int = 1            # 1 = literal single-market formula; K > 1 = time pseudo-markets
    # Part II
    propagate: bool = True        # False = ablation: final score is x0 (no network step)
    rho: float = 0.5
    tol: float = 1e-5
    # Algorithm 1
    theta: float = 0.9
    # Algorithm 2
    theta_lo: float = 0.8
    theta_hi: float = 0.99
    y_bar: float = 0.1
    zeta: float = 0.001

    def label(self) -> str:
        parts = [f"K={self.n_windows}"]
        if not self.propagate:
            parts.append("no-prop")
        if self.rho != 0.5:
            parts.append(f"rho={self.rho}")
        return ",".join(parts)


@dataclass
class Detection:
    config: DetectorConfig
    wallets: pd.DataFrame           # index wallet: x0, x, r, n_closures, volume, degree, flagged_*
    pairs: pd.DataFrame             # pair table + score, in_graph, flag_fixed, flag_star
    theta_star: float
    threshold_table: pd.DataFrame
    history: np.ndarray             # history[k] = x(k) over graph wallets (row 0 = x0)
    n_iter: int
    V: np.ndarray
    index: list
    stats: dict = field(default_factory=dict)


def detect(trades: pd.DataFrame, config: DetectorConfig = None, horizon=None) -> Detection:
    cfg = config or DetectorConfig()
    pairs = pair_trades(trades)
    index, V, self_qty = volume_graph(pairs)
    n = len(index)

    x0_tab = initial_scores(trades, c=cfg.c, n_windows=cfg.n_windows, horizon=horizon)
    x0 = x0_tab["x0"].reindex(index).fillna(0.0).to_numpy()

    if n == 0:
        x, hist, n_iter = np.zeros(0), np.zeros((1, 0)), 0
    elif cfg.propagate:
        x, hist, n_iter = propagate(x0, row_normalize(V), rho=cfg.rho, tol=cfg.tol)
    else:
        x, hist, n_iter = x0.copy(), x0[None, :], 0

    sel = select_threshold(x, V, cfg.theta_lo, cfg.theta_hi, cfg.y_bar, cfg.zeta) if n else \
        dict(theta_star=np.inf, table=pd.DataFrame(columns=["theta", "Y", "objective", "feasible"]),
             r=np.zeros(0))

    score = pd.Series(x, index=index)
    pairs = pairs.copy()
    pairs["in_graph"] = pairs["buyer"] != pairs["seller"]
    pairs["score"] = np.where(pairs["in_graph"],
                              np.minimum(pairs["buyer"].map(score), pairs["seller"].map(score)),
                              np.nan)
    pairs["flag_fixed"] = pairs["score"] >= cfg.theta
    pairs["flag_star"] = pairs["score"] >= sel["theta_star"]

    wallets = pd.DataFrame(index=pd.Index(index, name="wallet"))
    wallets["x0"] = x0
    wallets["x"] = x
    wallets["r"] = sel["r"]
    wallets = wallets.join(x0_tab[["n_closures", "volume", "n_clears", "n_closed_windows"]])
    wallets["degree"] = (V > 0).sum(axis=1)
    for col in ("flag_fixed", "flag_star"):
        hit = pairs.loc[pairs[col], ["buyer", "seller"]].to_numpy().ravel()
        wallets[col] = wallets.index.isin(set(hit))

    stats = dict(clear_stats(trades), self_pair_qty=self_qty, n_wallets=n,
                 n_edges=int((V > 0).sum() // 2), total_volume=float(trades["qty"].sum() / 2.0))
    return Detection(config=cfg, wallets=wallets, pairs=pairs, theta_star=sel["theta_star"],
                     threshold_table=sel["table"], history=hist, n_iter=n_iter, V=V,
                     index=index, stats=stats)


def detect_run(run_dir, config: DetectorConfig = None) -> Detection:
    """Detector on one exported run, reading observable data only."""
    obs = load_observable(run_dir)
    horizon = load_meta(run_dir)["config"]["sim_time"]
    return detect(obs["trades"], config, horizon=horizon)
