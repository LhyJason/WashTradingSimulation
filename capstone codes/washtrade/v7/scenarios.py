r"""scenarios.py -- v7 backgrounds and named wash scenarios.

Backgrounds are v6's (thin, thick, active, active_thin, active_dense). Scenarios: v6's thirteen
(identical definitions) plus v7 adversaries. The standard evasive group is a triad cycle drawn
from a pool of 12 wallets, with round size 1..4 and a residual on half of the rounds:

    triad_pool12      rotation only        k = 3 from a pool of 12, random draw
    triad_pool96      rotation only        pool of 96: almost every wallet trades in one round
    triad_size4       size only            Q uniform on 1..4, executed as unit events
    triad_block4      size only            same Q, one qty-Q event (visible on the tape)
    triad_res         residual only        p = 0.5, d = 1
    triad_evasive     all three            pool 12, Q 1..4, p = 0.5
    triad_evasive_g5  all three, gap 5
    evasive_camo      all three + camouflage of the whole pool, total rate 0.06

Camouflage on a pool: every pool wallet also trades as a ZI at rate CAMO_TOTAL / pool, so the
market gains the same ordinary activity whatever the pool size (a fixed triad: 0.02 per wallet)
and a wallet's ratio of ordinary to wash trading stays roughly constant as the pool grows.
"""
import random

import numpy as np
import torch

from washtrade.v6.scenarios import BACKGROUNDS, LAM_WASH, background_spec  # noqa: F401
from washtrade.v6.scenarios import SCENARIOS as V6_SCENARIOS

from .simulator import WashGroup, WashTradingSimulator

POOL = 12
Q_RANGE = (1, 4)
RESIDUAL_PROB = 0.5
CAMO_TOTAL = 0.06        # summed camouflage arrival rate of a pool


def _g(n_wallets, **kw):
    kw.setdefault("lam_wash", LAM_WASH)
    return WashGroup(n_wallets=n_wallets, **kw)


def evasive_group(pool=POOL, cycle_len=3, q_range=Q_RANGE, residual_prob=RESIDUAL_PROB, **kw):
    """The standard evasive group; every argument can be overridden."""
    if pool == cycle_len:
        kw.setdefault("rotation", "fixed")
    else:
        kw.setdefault("rotation", "random")
    return _g(pool, cycle_len=cycle_len, q_range=q_range, residual_prob=residual_prob, **kw)


def _as_v7(groups):
    return [WashGroup(**{k: getattr(g, k) for k in g.__dataclass_fields__}) for g in groups]


SCENARIOS = {name: (lambda f=f: _as_v7(f())) for name, f in V6_SCENARIOS.items()}
SCENARIOS.update({
    "triad_pool12":     lambda: [_g(POOL, cycle_len=3, rotation="random")],
    "triad_pool96":     lambda: [_g(96, cycle_len=3, rotation="random")],
    "triad_size4":      lambda: [_g(3, q_range=Q_RANGE)],
    "triad_block4":     lambda: [_g(3, q_range=Q_RANGE, size_exec="block")],
    "triad_res":        lambda: [_g(3, residual_prob=RESIDUAL_PROB)],
    "triad_evasive":    lambda: [evasive_group()],
    "triad_evasive_g5": lambda: [evasive_group(gap=5)],
    "evasive_camo":     lambda: [evasive_group(camouflage=True, camouflage_lam=CAMO_TOTAL / POOL)],
})


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_background(background):
    """Name or dict -> (name, spec dict)."""
    if isinstance(background, str):
        if background not in BACKGROUNDS:
            raise KeyError(f"unknown background {background!r}; known: {list(BACKGROUNDS)}")
        return background, dict(BACKGROUNDS[background])
    spec = dict(background)
    name = spec.pop("name", None) or "custom"
    return name, spec


def resolve_groups(scenario):
    """Scenario name, list of WashGroup / dicts -> list of v7 WashGroup."""
    if isinstance(scenario, str):
        if scenario not in SCENARIOS:
            raise KeyError(f"unknown scenario {scenario!r}; known: {list(SCENARIOS)}")
        return SCENARIOS[scenario]()
    return [g if isinstance(g, WashGroup) else WashGroup(**g) for g in scenario]


def build(background, scenario, seed, sim_time=2000, record_lob=True, lob_depth_levels=5):
    """Construct (not run) the simulator for one (background, scenario, seed)."""
    _, bg = resolve_background(background)
    groups = resolve_groups(scenario)
    comp = dict(bg.pop("composition"))
    seed_all(seed)
    return WashTradingSimulator(comp, wash_groups=groups, wash_seed=seed, sim_time=sim_time,
                                record_lob=record_lob, lob_depth_levels=lob_depth_levels, **bg)


def run(background, scenario, seed, **kw):
    sim = build(background, scenario, seed, **kw)
    sim.run()
    return sim


def run_id(background, scenario, seed):
    return f"{background}__{scenario}__s{int(seed):03d}"
