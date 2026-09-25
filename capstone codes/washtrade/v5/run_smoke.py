"""Smoke checks for v5 microprice-centered pricing."""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_CAP)
for _path in (_ROOT, _CAP):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from washtrade.v5 import scenarios


def main():
    sim = scenarios.run("thick", "triad", 0, record_lob=False)
    events = sim.controllers[0].event_log
    two_sided = [event for event in events
                 if math.isfinite(event["best_bid_at_open"])
                 and math.isfinite(event["best_ask_at_open"])]
    assert len(two_sided) > len(events) * 0.8
    for event in two_sided:
        bid_depth = event["bid_depth_at_open"]
        ask_depth = event["ask_depth_at_open"]
        expected = ((event["best_ask_at_open"] * bid_depth
                     + event["best_bid_at_open"] * ask_depth)
                    / (bid_depth + ask_depth))
        assert abs(event["microprice_anchor"] - expected) < 1e-8
    assert all(event["best_bid_at_open"] < event["target_price"]
               < event["best_ask_at_open"] for event in two_sided)
    assert all(event["sell_px"] == event["buy_px"] == event["target_price"]
               for event in events)
    assert all(abs(event["target_price"] / event["tick_size"]
                   - round(event["target_price"] / event["tick_size"])) < 1e-8
               for event in events)
    assert sum(abs(event["target_price"] - event["midpoint_anchor"]) <= 0.5
               for event in two_sided) < len(two_sided) * 0.1
    repeat = scenarios.run("thick", "triad", 0, record_lob=False)
    assert events == repeat.controllers[0].event_log
    gap = scenarios.run("thin", "triad_gap5", 0, record_lob=False)
    assert any(event["intercepted"] for event in gap.controllers[0].event_log)
    print("RESULT: ALL PASS")


if __name__ == "__main__":
    main()
