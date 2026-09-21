r"""lob.py -- LOBRecorder (v3): non-invasive observer of the limit order book.

Same design as v2 (see washtrade/v2/lob.py):
  * reads only the four-heap's unmatched queues and the ALREADY-generated fundamental values --
    never ``get_fundamental_value()``, because querying the lazy fundamental at a timestep
    nobody else asked for would consume torch draws differently and change the run;
  * one row per cleared timestep, ``*_pre`` = book inherited at the start of the step,
    unsuffixed = book after the clear.

v3 changes: wash accounting is by ORDER id (a camouflage wallet's ordinary ZI fills are not wash
legs), posts can come from several groups in one step, and ``export.py`` splits each row into an
observable part and a label part (``LABEL_COLUMNS``).
"""
import math

import pandas as pd

from .controller import is_wash_oid

SNAPSHOT_COLUMNS = [
    "time",
    "best_bid_pre", "best_ask_pre", "mid_pre", "spread_pre",
    "bid_depth_pre", "ask_depth_pre", "n_bid_orders_pre", "n_ask_orders_pre",
    "best_bid", "best_ask", "mid", "spread",
    "bid_depth", "ask_depth", "n_bid_orders", "n_ask_orders",
    "clear_price", "n_matched_legs", "volume",
    # --- latent / label columns (never exported to observable/) ---
    "fundamental", "fund_stale", "wash_legs", "wash_posts", "wash_phase",
]
LABEL_COLUMNS = ["fundamental", "fund_stale", "wash_legs", "wash_posts", "wash_phase"]


def _f(x):
    """+-inf -> NaN (so the columns stay numeric and plot cleanly)."""
    return float("nan") if (x is None or math.isinf(x)) else float(x)


class LOBRecorder:
    def __init__(self, depth_levels: int = 5, enabled: bool = True):
        self.depth_levels = int(depth_levels)
        self.enabled = bool(enabled)
        self.rows = []
        self.levels = []

    # -- book reading -------------------------------------------------------- #
    @staticmethod
    def _side(queue, is_bid: bool):
        by_price = {}
        total = 0.0
        n_orders = 0
        for order in queue.order_dict.values():
            p = float(order.price)
            q = float(order.quantity)
            if q <= 0:
                continue
            agg = by_price.setdefault(p, [0.0, 0])
            agg[0] += q
            agg[1] += 1
            total += q
            n_orders += 1
        levels = sorted(((p, v[0], v[1]) for p, v in by_price.items()),
                        key=lambda x: x[0], reverse=is_bid)
        best = levels[0][0] if levels else (-math.inf if is_bid else math.inf)
        return best, total, n_orders, levels

    def book_state(self, market):
        ob = market.order_book
        bb, bq, bn, blv = self._side(ob.buy_unmatched, is_bid=True)
        ba, aq, an, alv = self._side(ob.sell_unmatched, is_bid=False)
        two_sided = not (math.isinf(bb) or math.isinf(ba))
        return dict(best_bid=bb, best_ask=ba,
                    mid=0.5 * (bb + ba) if two_sided else math.nan,
                    spread=(ba - bb) if two_sided else math.nan,
                    bid_depth=bq, ask_depth=aq, n_bid_orders=bn, n_ask_orders=an,
                    bid_levels=blv, ask_levels=alv)

    @staticmethod
    def _fundamental_cached(market, t):
        fv = getattr(market.fundamental, "fundamental_values", None)
        if not fv:
            return math.nan, True
        if t in fv:
            return float(fv[t]), False
        prior = [k for k in fv if k <= t]
        if not prior:
            return math.nan, True
        return float(fv[max(prior)]), True

    # -- recording ----------------------------------------------------------- #
    def record(self, market, t, pre, new_matched, posts=(), lookup=None):
        """Append the snapshot for timestep t. ``posts`` = [(group_id, agent_id, Order), ...]
        injected this step; ``lookup(order_id)`` returns that wash order's metadata."""
        if not self.enabled:
            return
        post = self.book_state(market)
        fund, stale = self._fundamental_cached(market, t)

        phases = []
        if posts and lookup is not None:
            by_group = {}
            for gid, _, order in posts:
                meta = lookup(order.order_id) or {}
                by_group.setdefault(gid, set()).add(meta.get("leg"))
            for gid in sorted(by_group):
                legs = by_group[gid]
                ph = "both" if legs == {1, 2} else ("open" if legs == {1} else "close")
                phases.append(f"g{gid}:{ph}")

        self.rows.append(dict(
            time=t,
            best_bid_pre=_f(pre["best_bid"]), best_ask_pre=_f(pre["best_ask"]),
            mid_pre=_f(pre["mid"]), spread_pre=_f(pre["spread"]),
            bid_depth_pre=pre["bid_depth"], ask_depth_pre=pre["ask_depth"],
            n_bid_orders_pre=pre["n_bid_orders"], n_ask_orders_pre=pre["n_ask_orders"],
            best_bid=_f(post["best_bid"]), best_ask=_f(post["best_ask"]),
            mid=_f(post["mid"]), spread=_f(post["spread"]),
            bid_depth=post["bid_depth"], ask_depth=post["ask_depth"],
            n_bid_orders=post["n_bid_orders"], n_ask_orders=post["n_ask_orders"],
            clear_price=float(new_matched[0].price) if new_matched else math.nan,
            n_matched_legs=len(new_matched),
            volume=sum(float(mo.order.quantity) for mo in new_matched) / 2.0,
            fundamental=fund, fund_stale=stale,
            wash_legs=sum(1 for mo in new_matched if is_wash_oid(mo.order.order_id)),
            wash_posts=len(posts),
            wash_phase=";".join(phases),
        ))

        if self.depth_levels > 0:
            for side, key in (("bid", "bid_levels"), ("ask", "ask_levels")):
                for lvl, (p, q, n) in enumerate(post[key][:self.depth_levels]):
                    self.levels.append(dict(time=t, side=side, level=lvl,
                                            price=float(p), qty=float(q), n_orders=int(n)))

    # -- export -------------------------------------------------------------- #
    def snapshots_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=SNAPSHOT_COLUMNS)

    def levels_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.levels, columns=["time", "side", "level",
                                                  "price", "qty", "n_orders"])
