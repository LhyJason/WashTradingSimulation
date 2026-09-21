r"""export.py -- the dataset export schema: what a detector may see vs. the ground truth.

One run is written as

    <run_dir>/
      meta.json                 run id, scenario, full config, summary, schema version
      observable/               <- the ONLY folder a detector is allowed to read
        trades.csv              leg_id, time, wallet, side, qty, price
        lob.csv                 per-clear book snapshot (quotes, depth, clear price, volume)
        lob_levels.csv          top-K price levels per side per clear
      labels/                   <- ground truth, for evaluation only
        wallets.csv             wallet -> agent_id, behaviour type, is_wash, group, camouflage
        trade_legs.csv          leg_id -> order_id, is_wash_leg, group, event, leg number
        wash_events.csv         one row per wash event (intent + realized outcome)
        wash_fills.csv          one row per realized fill of a wash order
        wash_orders.csv         every wash order posted and how much of it filled
        lob_latent.csv          per-clear latent values: fundamental, wash legs/posts/phase

What a detector sees mirrors a public on-chain tape (the Polymarket setting of Sirolly et al.):
every fill with its account and side, but nothing a real exchange would not reveal. Three
leaks are closed on purpose:

  * **order ids are dropped** -- wash orders use ids >= 1e9, background ids are <= 1e7;
  * **wallets are anonymized** with a seeded permutation (``w000``, ``w001``, ...) -- raw agent
    ids put every wash wallet after the background agents;
  * **legs are shuffled within each clear** before ``leg_id`` is assigned -- the matched-order
    list is built from heap internals, whose order depends on insertion order.

The batch auction does not say who traded with whom inside a clear; reconstructing pairs from
``trades.csv`` is the detector's job (see detection/v1/pairing.py).

TODO: the column layout is a working default. If it changes before the dataset format is
frozen, bump SCHEMA_VERSION so old and new runs can be told apart.
"""
import json
import os

import numpy as np
import pandas as pd

from marketsim.fourheap.constants import SELL

from . import analysis as A
from .controller import is_wash_oid
from .lob import LABEL_COLUMNS, SNAPSHOT_COLUMNS

SCHEMA_VERSION = "washtrade.v3/1"
OBSERVABLE_TRADE_COLUMNS = ["leg_id", "time", "wallet", "side", "qty", "price"]
OBSERVABLE_LOB_COLUMNS = [c for c in SNAPSHOT_COLUMNS if c not in LABEL_COLUMNS]

_LEG_SHUFFLE_OFFSET = 101
_WALLET_PERM_OFFSET = 211


def wallet_codes(sim, seed, anonymize=True) -> dict:
    """agent_id -> public wallet code. Seeded, so re-exporting a run gives the same codes."""
    ids = sorted(sim.agents)
    if not anonymize:
        return {aid: f"w{aid:03d}" for aid in ids}
    perm = np.random.default_rng(int(seed) + _WALLET_PERM_OFFSET).permutation(len(ids))
    return {aid: f"w{int(p):03d}" for aid, p in zip(ids, perm)}


def _behaviour(sim, aid):
    t = sim.agent_types[aid]
    if aid in sim.camouflage_ids:
        g = sim.groups[sim.wallet_group[aid]]
        return f"{t}+{g.camouflage_type}"
    return t


def export_run(sim, run_dir, *, run_id, seed, scenario=None, background=None,
               anonymize=True, save_levels=True, extra_meta=None) -> dict:
    """Write one simulated run in the layout described above. Returns {relative file name: path}."""
    obs_dir = os.path.join(run_dir, "observable")
    lab_dir = os.path.join(run_dir, "labels")
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    paths = {}

    def _dump(folder, name, df):
        p = os.path.join(folder, f"{name}.csv")
        df.to_csv(p, index=False)
        paths[f"{os.path.basename(folder)}/{name}.csv"] = p

    codes = wallet_codes(sim, seed, anonymize)

    # ---- legs: shuffle within each clear, then number them ------------------------------ #
    legs = A.trade_log_df(sim)
    rng = np.random.default_rng(int(seed) + _LEG_SHUFFLE_OFFSET)
    legs["_r"] = rng.random(len(legs))
    legs = legs.sort_values(["time", "_r"], kind="mergesort").reset_index(drop=True)
    legs["leg_id"] = np.arange(len(legs))
    legs["wallet"] = legs["agent_id"].map(codes)

    _dump(obs_dir, "trades", legs[OBSERVABLE_TRADE_COLUMNS])
    lob = A.lob_df(sim)
    _dump(obs_dir, "lob", lob[OBSERVABLE_LOB_COLUMNS])
    if save_levels:
        _dump(obs_dir, "lob_levels", A.lob_levels_df(sim))

    # ---- labels --------------------------------------------------------------------------- #
    vol = legs.groupby("agent_id")["qty"].sum()
    n_legs = legs.groupby("agent_id").size()
    wallets = pd.DataFrame([dict(
        wallet=codes[aid], agent_id=aid, behaviour=_behaviour(sim, aid),
        is_wash=aid in sim.wallet_group, group_id=sim.wallet_group.get(aid, -1),
        camouflage=aid in sim.camouflage_ids,
        final_position=int(sim.agents[aid].position), cash=float(sim.agents[aid].cash),
        n_legs=int(n_legs.get(aid, 0)), volume=float(vol.get(aid, 0.0)),
    ) for aid in sorted(sim.agents)]).sort_values("wallet").reset_index(drop=True)
    _dump(lab_dir, "wallets", wallets)

    _dump(lab_dir, "trade_legs", legs[["leg_id", "wallet", "agent_id", "order_id", "is_wash_leg",
                                       "group_id", "event_idx"]])

    ev = A.event_log_df(sim)
    if len(ev):
        ev["seller_wallet"] = ev["seller"].map(codes)
        ev["buyer_wallet"] = ev["buyer"].map(codes)
    _dump(lab_dir, "wash_events", ev)
    fills = A.fill_log_df(sim)
    fills["wallet"] = fills["agent_id"].map(codes)
    _dump(lab_dir, "wash_fills", fills)
    orders = A.order_log_df(sim)
    if len(orders):
        orders["wallet"] = orders["agent_id"].map(codes)
    _dump(lab_dir, "wash_orders", orders)
    _dump(lab_dir, "lob_latent", lob[["time"] + LABEL_COLUMNS])

    # ---- meta ----------------------------------------------------------------------------- #
    meta = dict(
        schema_version=SCHEMA_VERSION, run_id=run_id, seed=int(seed),
        scenario=scenario, background=background, anonymized=bool(anonymize),
        config=getattr(sim, "run_config", {}),
        summary=A.summarize(sim),
        groups=A.group_summary(sim).to_dict(orient="records"),
        observable_columns=dict(trades=OBSERVABLE_TRADE_COLUMNS, lob=OBSERVABLE_LOB_COLUMNS),
        files=sorted(paths),
    )
    if extra_meta:
        meta.update(extra_meta)
    p = os.path.join(run_dir, "meta.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    paths["meta.json"] = p
    return paths


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_observable(run_dir) -> dict:
    """What a detector gets: {'trades', 'lob', 'lob_levels'(if present)} DataFrames."""
    obs = os.path.join(run_dir, "observable")
    out = {"trades": pd.read_csv(os.path.join(obs, "trades.csv")),
           "lob": pd.read_csv(os.path.join(obs, "lob.csv"))}
    lv = os.path.join(obs, "lob_levels.csv")
    if os.path.exists(lv):
        out["lob_levels"] = pd.read_csv(lv)
    return out


def load_labels(run_dir) -> dict:
    """Ground truth for evaluation: every labels/*.csv as a DataFrame (empty files -> empty)."""
    lab = os.path.join(run_dir, "labels")
    out = {}
    for f in sorted(os.listdir(lab)):
        if f.endswith(".csv"):
            p = os.path.join(lab, f)
            try:
                out[f[:-4]] = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                out[f[:-4]] = pd.DataFrame()
    return out


def load_meta(run_dir) -> dict:
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as fh:
        return json.load(fh)
