"""Named v5 scenarios using the selected microprice pricing defaults."""

import random

import numpy as np
import torch

from .simulator import WashGroup, WashTradingSimulator

LAM_WASH = 0.05
LAM_WASH_LOW = 0.01

BACKGROUNDS = {
    "thin": dict(composition={"ZI": 13}, lam=5e-3),
    "thick": dict(composition={"ZI": 30}, lam=2e-2),
}


def _g(n_wallets, **kwargs):
    kwargs.setdefault("lam_wash", LAM_WASH)
    return WashGroup(n_wallets=n_wallets, **kwargs)


SCENARIOS = {
    "control": lambda: [],
    "triad": lambda: [_g(3)],
    "dyad": lambda: [_g(2)],
    "triad_gap5": lambda: [_g(3, gap=5)],
    "triad_gap20": lambda: [_g(3, gap=20)],
    "triad_camo": lambda: [_g(3, camouflage=True)],
    "triad_lowint": lambda: [_g(3, lam_wash=LAM_WASH_LOW)],
    "two_groups": lambda: [_g(3), _g(2)],
    "hard": lambda: [_g(3, gap=5, camouflage=True),
                     _g(2, gap=5, camouflage=True)],
    # Ablations for attributing any change to the center or to the range.
    "triad_midcenter": lambda: [_g(3, use_microprice=False)],
    "triad_narrow": lambda: [_g(3, price_range_fraction=0.2)],
    "triad_wide": lambda: [_g(3, price_range_fraction=0.8)],
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(background, scenario, seed, sim_time=2000, record_lob=True,
          lob_depth_levels=5):
    bg = BACKGROUNDS[background]
    seed_all(seed)
    return WashTradingSimulator(
        dict(bg["composition"]), wash_groups=SCENARIOS[scenario](), wash_seed=seed,
        sim_time=sim_time, lam=bg["lam"], record_lob=record_lob,
        lob_depth_levels=lob_depth_levels,
    )


def run(background, scenario, seed, **kwargs):
    sim = build(background, scenario, seed, **kwargs)
    sim.run()
    return sim


def run_id(background, scenario, seed):
    return f"{background}__{scenario}__s{int(seed):03d}"
