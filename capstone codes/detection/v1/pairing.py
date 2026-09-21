r"""pairing.py -- turning a batch-auction tape into counterparty pairs.

The problem: Polymarket's tape records maker and taker for every fill, so the paper's matrix B
comes straight from the data. PyMarketSim clears a uniform-price call auction: every buy leg
matched at time t trades against every sell leg matched at t at one price, and the matched-order
list carries no pairing.

Rule used (pro-rata): within one clear with buy legs b_1..b_m and sell legs s_1..s_n
(sum b = sum s = Q), leg pair (b_k, s_l) is assigned volume  b_k * s_l / Q.

Why this rule:
  * it is exact whenever one side of the clear has a single wallet -- which in this platform is
    almost every clear (calibration: <= 2% of clears have more than one wallet on a side);
  * it is the unique rule that treats all legs in a clear symmetrically (no invented priority)
    and it conserves each leg's quantity;
  * it never fabricates an edge between two wallets that were not in the same clear.

``clear_stats`` reports how much volume sat in genuinely ambiguous clears so the approximation
can be quoted alongside every result.

TODO: pro-rata is a working default. Re-check it if the detection results turn out to be
sensitive to how ambiguous (many-vs-many) clears are split.
"""
import numpy as np
import pandas as pd

PAIR_COLUMNS = ["time", "buyer", "seller", "qty", "price", "buy_leg_id", "sell_leg_id"]


def pair_trades(trades: pd.DataFrame, check: bool = True) -> pd.DataFrame:
    """observable trades (leg_id, time, wallet, side, qty, price) -> one row per (buy leg, sell leg)
    pair within each clear, with pro-rata volume."""
    if trades.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    buys = trades.loc[trades["side"] == "BUY", ["time", "leg_id", "wallet", "qty", "price"]]
    sells = trades.loc[trades["side"] == "SELL", ["time", "leg_id", "wallet", "qty"]]
    qb = buys.groupby("time")["qty"].sum()
    qs = sells.groupby("time")["qty"].sum()
    if check:
        both = pd.concat([qb.rename("b"), qs.rename("s")], axis=1).fillna(0.0)
        bad = both[(both["b"] - both["s"]).abs() > 1e-9]
        if len(bad):
            raise ValueError(f"buy and sell quantity differ in {len(bad)} clears, e.g. t={bad.index[0]}")

    m = buys.merge(sells, on="time", suffixes=("_b", "_s"))
    out = pd.DataFrame(dict(
        time=m["time"], buyer=m["wallet_b"], seller=m["wallet_s"],
        qty=m["qty_b"] * m["qty_s"] / m["time"].map(qb),
        price=m["price"], buy_leg_id=m["leg_id_b"], sell_leg_id=m["leg_id_s"],
    ))
    return out.sort_values(["time", "buy_leg_id", "sell_leg_id"]).reset_index(drop=True)


def clear_stats(trades: pd.DataFrame) -> dict:
    """How approximate the pro-rata pairing is for this tape."""
    if trades.empty:
        return dict(n_clears=0, n_ambiguous_clears=0, ambiguous_volume_share=float("nan"))
    per = trades.groupby(["time", "side"])["wallet"].nunique().unstack(fill_value=0)
    amb_times = per.index[(per.get("BUY", 0) > 1) & (per.get("SELL", 0) > 1)]
    vol = trades.groupby("time")["qty"].sum()
    return dict(
        n_clears=int(len(per)),
        n_ambiguous_clears=int(len(amb_times)),
        ambiguous_volume_share=float(vol.reindex(amb_times).sum() / vol.sum()) if vol.sum() else float("nan"),
    )


def volume_graph(pairs: pd.DataFrame, drop_self: bool = True):
    """Symmetric counterparty volume matrix.

    Returns (wallets, V, self_qty): ``wallets`` is the sorted node list, ``V[i, j]`` the total
    volume wallets i and j traded with each other in either direction (so V = V.T, zero diagonal),
    ``self_qty`` the volume of pairs where one wallet sat on both sides of a clear -- dropped,
    as B has a zero diagonal (and Polymarket blocks self-trades outright)."""
    self_mask = pairs["buyer"] == pairs["seller"]
    self_qty = float(pairs.loc[self_mask, "qty"].sum())
    p = pairs.loc[~self_mask] if drop_self else pairs
    wallets = sorted(set(p["buyer"]) | set(p["seller"]))
    idx = {w: i for i, w in enumerate(wallets)}
    V = np.zeros((len(wallets), len(wallets)))
    np.add.at(V, (p["buyer"].map(idx).to_numpy(), p["seller"].map(idx).to_numpy()), p["qty"].to_numpy())
    V = V + V.T
    return wallets, V, self_qty
