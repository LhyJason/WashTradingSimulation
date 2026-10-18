r"""analysis.py -- v7 run metrics: v6's, plus what the evasive adversary changes.

Everything in ``washtrade.v6.analysis`` works unchanged on a v7 simulator and is re-exported.
v7 adds, per wash group:

* **rotation** -- how many pool wallets ever trade wash, rounds per used wallet, and how often
  the same (unordered) pair of wallets trades wash: ``n_pairs``, ``events_per_pair_mean/max``.
  A fixed triad has 3 pairs carrying every event; a large pool spreads them thin.
* **size** -- mean round size and mean quantity of a wash leg (``> 1`` only with "block").
* **residual** -- share of rounds whose closing edge was off, and the realized result on the
  tape: per used wallet, how often its net position (all its legs, as a detector sees it) returns
  to exactly zero, and, at the end of every completed round, the share of the round's members
  whose realized position is not zero (``round_end_nonzero_share``) and its mean |position|.
  Round ends rather than the end of the run: a run usually stops in the middle of a round.
"""
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from marketsim.fourheap.constants import SELL

from washtrade.v3.controller import is_wash_oid
from washtrade.v6.analysis import (  # noqa: F401  (re-exported)
    background_activity, cycle_boundary_nets, event_log_df, fill_log_df, group_leakage,
    lob_dense, lob_df, lob_levels_df, market_quality, order_log_df, position_timeseries,
    reconstruct_positions, trade_log_df,
)
from washtrade.v6.analysis import group_summary as _v6_group_summary
from washtrade.v6.analysis import summarize as _v6_summarize


CLEAR_TIME_OFFSET = 1    # matched_orders time = simulator step + 1


def _rate(num, den):
    return (num / den) if den else float("nan")


def round_log_df(sim) -> pd.DataFrame:
    rows = []
    for c in sim.controllers:
        for r in c.round_log:
            rows.append(dict(r, members=";".join(str(m) for m in r["members"])))
    return pd.DataFrame(rows)


def _position_path(sim):
    """[(time, {agent_id: net position after that clear, for agents that traded in it})]."""
    pos = defaultdict(int)
    by_time = defaultdict(lambda: defaultdict(int))
    for mo in sim.markets[0].matched_orders:
        o = mo.order
        by_time[mo.time][o.agent_id] += (-o.quantity) if o.order_type == SELL else o.quantity
    path = []
    for t in sorted(by_time):
        for aid, dq in by_time[t].items():
            pos[aid] += dq
        path.append((t, {aid: pos[aid] for aid in by_time[t]}))
    return path


def zero_returns(sim) -> dict:
    """agent_id -> number of times its net position (all legs, per clear) returns to exactly 0
    from a non-zero value. This is the event a closure-based detector counts."""
    pos = defaultdict(int)
    count = defaultdict(int)
    for _, step in _position_path(sim):
        for aid, p in step.items():
            if pos[aid] != 0 and p == 0:
                count[aid] += 1
            pos[aid] = p
    return dict(count)


def round_end_positions(sim) -> list:
    """[(group_id, round_idx, agent_id, realized net position)] for every member of every round
    that completed, right after the clear of the round's last event. Residuals, failed legs and
    camouflage trades all show up here; a run that stops mid-round does not."""
    ends = {}
    for c in sim.controllers:
        done = Counter(e.get("round_idx") for e in c.event_log)
        for r in c.round_log:
            if "close_time" in r and done[r["round_idx"]] == r["n_events"]:
                # the clear of simulator step t is stamped t + 1 on the tape (upstream
                # ``market.step`` advances its clock before clearing)
                ends.setdefault(r["close_time"] + CLEAR_TIME_OFFSET, []).append((c.group_id, r))
    pos = defaultdict(int)
    out = []
    pending = sorted(ends)

    def flush(upto):                             # rounds that ended at or before ``upto``
        while pending and pending[0] <= upto:
            for gid, r in ends[pending.pop(0)]:
                out.extend((gid, r["round_idx"], m, pos[m]) for m in r["members"])

    for t, step in _position_path(sim):
        flush(t - 1)                             # ended in a step without any clear
        pos.update(step)
        flush(t)
    flush(float("inf"))
    return out


def _pair_counts(c) -> dict:
    """{frozenset(seller, buyer): number of wash events on that pair} for one controller."""
    out = defaultdict(int)
    for e in c.event_log:
        out[frozenset((e["seller"], e["buyer"]))] += 1
    return out


def group_summary(sim) -> pd.DataFrame:
    df = _v6_group_summary(sim)
    if df.empty:
        return df
    zr = zero_returns(sim)
    ends = defaultdict(list)
    for gid, _, _, p in round_end_positions(sim):
        ends[gid].append(p)
    legs = defaultdict(list)
    for mo in sim.markets[0].matched_orders:
        if is_wash_oid(mo.order.order_id):
            k = sim.wash_meta(mo.order.order_id)["group_id"]
            legs[k].append(float(mo.order.quantity))
    rows = []
    for c in sim.controllers:
        used = [w for w in c.wallet_ids if c.rounds_used[w] > 0]
        pairs = _pair_counts(c)
        n_ev = len(c.event_log)
        rl = c.round_log
        rows.append(dict(
            pool_size=len(c.wallet_ids), cycle_len=c.cycle_len, rotation=c.rotation,
            q_lo=(c.q_range[0] if c.q_range else 1), q_hi=(c.q_range[1] if c.q_range else 1),
            size_exec=c.size_exec, residual_prob=c.residual_prob, residual_max=c.residual_max,
            n_rounds=len(rl),
            n_wallets_used=len(used),
            rounds_per_used_wallet=_rate(sum(c.rounds_used[w] for w in used), len(used)),
            n_pairs=len(pairs),
            n_possible_pairs=len(list(combinations(c.wallet_ids, 2))),
            events_per_pair_mean=_rate(n_ev, len(pairs)),
            events_per_pair_max=max(pairs.values()) if pairs else 0,
            mean_round_size=float(np.mean([r["size"] for r in rl])) if rl else float("nan"),
            mean_wash_leg_qty=float(np.mean(legs[c.group_id])) if legs[c.group_id] else float("nan"),
            residual_round_share=_rate(sum(r["residual"] != 0 for r in rl), len(rl)),
            zero_returns_per_used_wallet=_rate(sum(zr.get(w, 0) for w in used), len(used)),
            round_end_nonzero_share=_rate(sum(p != 0 for p in ends[c.group_id]),
                                          len(ends[c.group_id])),
            round_end_abs_position=_rate(sum(abs(p) for p in ends[c.group_id]),
                                         len(ends[c.group_id])),
        ))
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


def summarize(sim) -> dict:
    """v6's scalar metrics + honest-wallet closures (the detector's false-positive raw material)
    + pooled v7 group metrics."""
    s = _v6_summarize(sim)
    zr = zero_returns(sim)
    wallets = set(sim.wash_wallet_ids)
    honest = [a for a in sim.agents if a not in wallets]
    s["honest_zero_returns_per_agent"] = _rate(sum(zr.get(a, 0) for a in honest), len(honest))
    s["honest_ever_zero_share"] = _rate(sum(zr.get(a, 0) > 0 for a in honest), len(honest))
    g = group_summary(sim)
    if len(g):
        for col in ("n_rounds", "n_wallets_used", "n_pairs", "events_per_pair_mean",
                    "events_per_pair_max", "residual_round_share",
                    "zero_returns_per_used_wallet", "round_end_nonzero_share",
                    "round_end_abs_position"):
            s[col] = float(g[col].mean()) if col != "n_rounds" else int(g[col].sum())
    return s
