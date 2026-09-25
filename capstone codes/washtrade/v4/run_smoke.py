"""Check randomized wash prices, crossing, reproducibility, and v3 baseline parity."""

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_CAP)
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from washtrade.v3 import scenarios as V3
from washtrade.v4 import scenarios as V4
from washtrade.v4.export import export_run, load_labels, load_meta, load_observable


def _tape(sim):
    return [(m.time, m.order.agent_id, m.order.order_id, m.order.order_type,
             m.order.quantity, m.price) for m in sim.markets[0].matched_orders]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    baseline = V3.run("thin", "triad", 0)
    midpoint = V4.run("thin", "triad_midpoint", 0)
    assert _tape(baseline) == _tape(midpoint)
    assert all(a.position == midpoint.agents[aid].position and
               a.cash == midpoint.agents[aid].cash
               for aid, a in baseline.agents.items())
    print("[PASS] zero-width v4 reproduces v3 trades and accounts")

    sim = V4.run("thin", "triad", 0)
    events = sim.controllers[0].event_log
    assert len(events) > 20
    assert any(abs(e["price_offset"]) > 1e-6 for e in events)
    assert any(e["price_offset"] < 0 for e in events)
    assert any(e["price_offset"] > 0 for e in events)
    assert any(e["price_width"] > 1e-6 for e in events)
    assert all(e["sell_px"] <= e["buy_px"] for e in events)
    assert all(e["midpoint_anchor"] - e["price_radius"] - 1e-8 <= e["sell_px"]
               <= e["buy_px"] <= e["midpoint_anchor"] + e["price_radius"] + 1e-8
               for e in events)
    assert all(e["best_bid_at_open"] < e["sell_px"]
               and e["buy_px"] < e["best_ask_at_open"]
               for e in events if np.isfinite(e["best_bid_at_open"])
               and np.isfinite(e["best_ask_at_open"]))
    print("[PASS] two varied limits stay within the spread and cross")

    repeat = V4.run("thin", "triad", 0)
    assert _tape(sim) == _tape(repeat)
    assert sim.controllers[0].event_log == repeat.controllers[0].event_log
    print("[PASS] fixed seed reproduces events and fills")

    gap = V4.run("thin", "triad_gap5", 0)
    assert all(e["close_time"] - e["open_time"] == 5
               for e in gap.controllers[0].event_log if e["closed"])
    assert any(e["intercepted"] for e in gap.controllers[0].event_log)
    print("[PASS] exposed first legs still permit interception")

    with tempfile.TemporaryDirectory() as tmp:
        export_run(sim, tmp, run_id="thin__triad__s000", seed=0)
        obs = load_observable(tmp)
        lab = load_labels(tmp)
        meta = load_meta(tmp)
        assert meta["schema_version"] == "washtrade.v4/1"
        assert "price_offset" in lab["wash_events"]
        assert "price_width" in lab["wash_events"]
        assert "order_id" not in obs["trades"]
        assert len(obs["trades"]) == len(_tape(sim))
    print("[PASS] v4 export records price draws without leaking labels")
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
