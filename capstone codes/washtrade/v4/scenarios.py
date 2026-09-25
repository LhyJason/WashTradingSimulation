r"""scenarios.py -- randomized-price conditions of the v4 dataset.

A run = BACKGROUND (the honest market) x SCENARIO (which wash groups are present) x seed.
Scenarios are a short list of contrasts rather than a full factorial grid (capstone scope:
"no topology sweep, keep it simple"); each one changes one thing relative to ``triad``:

    control        no wash trading at all                -> false-positive baseline
    triad          1 group, 3 wallets, gap 0             -> the v1/v2 reference condition
    dyad           2 wallets instead of 3                -> single reciprocal edge
    triad_gap5     gap = 5                               -> some interception
    triad_gap20    gap = 20                              -> heavy interception
    triad_camo     wallets also trade as ZIs             -> "commingling" camouflage
    triad_lowint   lam_wash 0.01 instead of 0.05         -> wash is a smaller share of volume
    two_groups     a triad and a dyad, independent       -> several clusters in one market
    hard           two_groups + camouflage + gap 5       -> all complications at once

Backgrounds:
    thin     {"ZI": 13}, lam = 5e-3   PyMarketSim Table-3 setting used in v1/v2 (~15 honest trades)
    thick    {"ZI": 30}, lam = 2e-2   ~3.5x as many honest trades, wash share roughly halves
"""
import random

import numpy as np
import torch

from .simulator import WashGroup, WashTradingSimulator

LAM_WASH = 0.05          # standard wash intensity: one event every ~20 steps
LAM_WASH_LOW = 0.01

BACKGROUNDS = {
    "thin":  dict(composition={"ZI": 13}, lam=5e-3),
    "thick": dict(composition={"ZI": 30}, lam=2e-2),
}


def _g(n_wallets, **kw):
    kw.setdefault("lam_wash", LAM_WASH)
    return WashGroup(n_wallets=n_wallets, **kw)


SCENARIOS = {
    "control":      lambda: [],
    "triad":        lambda: [_g(3)],
    "triad_midpoint": lambda: [_g(3, price_range_fraction=0)],
    "triad_narrow": lambda: [_g(3, price_range_fraction=0.2)],
    "triad_wide":   lambda: [_g(3, price_range_fraction=0.8)],
    "dyad":         lambda: [_g(2)],
    "triad_gap5":   lambda: [_g(3, gap=5)],
    "triad_gap20":  lambda: [_g(3, gap=20)],
    "triad_camo":   lambda: [_g(3, camouflage=True)],
    "triad_lowint": lambda: [_g(3, lam_wash=LAM_WASH_LOW)],
    "two_groups":   lambda: [_g(3), _g(2)],
    "hard":         lambda: [_g(3, gap=5, camouflage=True), _g(2, gap=5, camouflage=True)],
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(background, scenario, seed, sim_time=2000, record_lob=True, lob_depth_levels=5):
    """Construct (not run) the simulator for one (background, scenario, seed)."""
    if background not in BACKGROUNDS:
        raise KeyError(f"unknown background {background!r}; known: {list(BACKGROUNDS)}")
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; known: {list(SCENARIOS)}")
    bg = BACKGROUNDS[background]
    seed_all(seed)
    return WashTradingSimulator(
        dict(bg["composition"]), wash_groups=SCENARIOS[scenario](), wash_seed=seed,
        sim_time=sim_time, lam=bg["lam"],
        record_lob=record_lob, lob_depth_levels=lob_depth_levels,
    )


def run(background, scenario, seed, **kw):
    sim = build(background, scenario, seed, **kw)
    sim.run()
    return sim


def run_id(background, scenario, seed):
    return f"{background}__{scenario}__s{int(seed):03d}"
