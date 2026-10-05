r"""scenarios.py -- v6 backgrounds and named wash scenarios.

A run = BACKGROUND (the honest market) x wash GROUPS x seed. ``build`` accepts either a named
background / scenario or explicit dicts, so the sweeps in ``design.py`` can vary any factor.

Backgrounds (all backgrounds are ZI-only; an HBL background is left for a later version):

    thin           {"ZI": 13}, lam 5e-3, r 0.01   = v3 thin  (PyMarketSim Table 3)
    thick          {"ZI": 30}, lam 2e-2, r 0.01   = v3 thick
    active         {"ZI": 30}, lam 2e-2, r 1e-3   thick with slow mean reversion
    active_thin    {"ZI": 13}, lam 5e-3, r 1e-3   thin  with slow mean reversion
    active_dense   {"ZI": 60}, lam 2e-2, r 1e-3   twice as many ZIs

Why r: a ZI values the asset at its forecast of the FINAL fundamental,
(1 - rho_t) * mean + rho_t * f_t with rho_t = (1 - r) ** (T - t). At r = 0.01 and T = 2000,
rho_t ~ 0 until the last few hundred steps, so after the opening burst every ZI values the asset
at the same constant and honest trading stops; more ZIs or faster arrivals only enlarge the
opening and closing bursts. At r = 1e-3 honest trades are spread over the whole run (20 seeds:
{"ZI": 30}, lam 2e-2 gives 21 honest legs per ZI instead of 4.3, and the triad's wash share falls
from 0.59 to 0.22).

Scenarios: v3's nine (identical definitions) plus v6 camouflage-intensity contrasts.
"""
import random

import numpy as np
import torch

from .simulator import WashGroup, WashTradingSimulator

LAM_WASH = 0.05          # standard wash intensity: one event every ~20 steps
LAM_WASH_LOW = 0.01
CAMO_LAM_HIGH = 0.05     # camouflage wallets arrive 2.5x as often as a thick/active ZI

BACKGROUNDS = {
    "thin":         dict(composition={"ZI": 13}, lam=5e-3, r=0.01),
    "thick":        dict(composition={"ZI": 30}, lam=2e-2, r=0.01),
    "active":       dict(composition={"ZI": 30}, lam=2e-2, r=1e-3),
    "active_thin":  dict(composition={"ZI": 13}, lam=5e-3, r=1e-3),
    "active_dense": dict(composition={"ZI": 60}, lam=2e-2, r=1e-3),
}


def background_spec(n_zi, lam, r=0.01, **extra):
    """An explicit background, in the same form as the entries of BACKGROUNDS."""
    return dict(composition={"ZI": int(n_zi)}, lam=float(lam), r=float(r), **extra)


def _g(n_wallets, **kw):
    kw.setdefault("lam_wash", LAM_WASH)
    return WashGroup(n_wallets=n_wallets, **kw)


SCENARIOS = {
    # --- v3's nine, unchanged ---
    "control":        lambda: [],
    "triad":          lambda: [_g(3)],
    "dyad":           lambda: [_g(2)],
    "triad_gap5":     lambda: [_g(3, gap=5)],
    "triad_gap20":    lambda: [_g(3, gap=20)],
    "triad_camo":     lambda: [_g(3, camouflage=True)],
    "triad_lowint":   lambda: [_g(3, lam_wash=LAM_WASH_LOW)],
    "two_groups":     lambda: [_g(3), _g(2)],
    "hard":           lambda: [_g(3, gap=5, camouflage=True), _g(2, gap=5, camouflage=True)],
    # --- v6 ---
    "triad_camo_hi":  lambda: [_g(3, camouflage=True, camouflage_lam=CAMO_LAM_HIGH)],
    "triad_camo_one": lambda: [_g(3, camouflage=True, n_camouflage=1, camouflage_lam=CAMO_LAM_HIGH)],
    "triad_micro":    lambda: [_g(3, gap=5, price_rule="microprice")],
    "hard_hi":        lambda: [_g(3, gap=5, camouflage=True, camouflage_lam=CAMO_LAM_HIGH),
                               _g(2, gap=5, camouflage=True, camouflage_lam=CAMO_LAM_HIGH)],
}


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
    """Scenario name, list of WashGroup / dicts -> list of WashGroup."""
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
