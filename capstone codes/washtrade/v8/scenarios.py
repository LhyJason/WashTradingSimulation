r"""scenarios.py -- v8 backgrounds, named scenarios, and the build/run helpers.

Backgrounds: v6's five, plus a multi-unit version of four of them. ``<bg>_mu4`` is <bg> with
honest order sizes uniform on {1..4}. Q_RANGE is (1, 4) too, so a wash leg of up to 4 units no
longer stands out by its size.

Scenarios: v7's (v6's nine + v7's eight, unchanged), plus the v8 execution variants of the
standard evasive group (pool 96, cycles of 3, Q in 1..4, residual on half of the rounds):

    evasive96           v7 execution (sequential edges, one round at a time)
    evasive96_inter     interleaved edges
    evasive96_pipe8     8 rounds open at once
    evasive96_ip8       both
    evasive96_block     one block order of Q units per edge (only sensible on a _mu background)
"""
import random

import numpy as np
import torch

from washtrade.v6.scenarios import BACKGROUNDS as V6_BACKGROUNDS
from washtrade.v6.scenarios import LAM_WASH
from washtrade.v7 import scenarios as S7

from .simulator import WashGroup, WashTradingSimulator

POOL = 96
Q_RANGE = S7.Q_RANGE
RESIDUAL_PROB = S7.RESIDUAL_PROB
CAMO_TOTAL = S7.CAMO_TOTAL
HONEST_SIZE = 4
MU_BASES = ("thin", "thick", "active", "active_dense")

BACKGROUNDS = dict(V6_BACKGROUNDS)
BACKGROUNDS.update({f"{b}_mu{HONEST_SIZE}": dict(V6_BACKGROUNDS[b], order_size_max=HONEST_SIZE)
                    for b in MU_BASES})


def _g(n_wallets, **kw):
    kw.setdefault("lam_wash", LAM_WASH)
    return WashGroup(n_wallets=n_wallets, **kw)


def evasive_group(pool=POOL, cycle_len=3, q_range=Q_RANGE, residual_prob=RESIDUAL_PROB, **kw):
    """The standard evasive group of v7, with v8's knobs available; every argument can be set."""
    kw.setdefault("rotation", "fixed" if pool == cycle_len else "random")
    return _g(pool, cycle_len=cycle_len, q_range=q_range, residual_prob=residual_prob, **kw)


def _as_v8(groups):
    return [WashGroup(**{k: getattr(g, k) for k in g.__dataclass_fields__}) for g in groups]


SCENARIOS = {name: (lambda f=f: _as_v8(f())) for name, f in S7.SCENARIOS.items()}
SCENARIOS.update({
    "evasive96":        lambda: [evasive_group()],
    "evasive96_inter":  lambda: [evasive_group(edge_order="interleaved")],
    "evasive96_pipe8":  lambda: [evasive_group(pipeline=8)],
    "evasive96_ip8":    lambda: [evasive_group(edge_order="interleaved", pipeline=8)],
    "evasive96_block":  lambda: [evasive_group(size_exec="block")],
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
    """Scenario name, list of WashGroup / dicts -> list of v8 WashGroup."""
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
