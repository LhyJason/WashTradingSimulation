r"""design.py -- the v8 experimental designs (sweeps), as flat lists of run specifications.

Same machinery as v7 (``RunSpec``, deterministic order, round-robin shards). Default 200 seeds.

    sweep       factors (x seeds)                                                    cells  runs @200
    adversary   4 bg x gap {0,5} x edge order {sequential, interleaved}
                x pipeline {1,4,8} x lam_wash {0.05, 0.0125}  + one control per bg     100    20000
    honest      4 bg x honest order size {1, 1..4}
                x {control, fixed triad, evasive, evasive block, evasive interleaved+pipeline 8}
                at gap 0                                                                 40     8000
                                                                                total   140    28000

"4 bg" = thin, thick, active, active_dense. In ``honest``, order size 1..4 means the ``<bg>_mu4``
background (scenarios.BACKGROUNDS), whose name differs from <bg>'s, so a detector's control
calibration keeps the two apart.

The wash group in both sweeps is v7's full evasive adversary: a pool of 96 wallets, cycles of 3
drawn at random, Q uniform on 1..4, a residual on half of the rounds, and camouflage at a total
rate of 0.06 over the pool. ``adversary`` varies v8's two execution knobs and the wash intensity.
lam_wash = 0.0125 is a quarter of the standard rate: positions are held four times longer, at a
quarter of the wash volume. ``honest`` crosses the honest order size with v7 execution, block
execution (one order of Q units per edge, which a multi-unit background can hide) and v8's
strongest execution. The fixed triad there is the standard v3-v7 triad (Q = 1, no camouflage).
"""
from dataclasses import asdict
from itertools import product

from washtrade.v6.design import _fmt
from washtrade.v7.design import RunSpec, shard  # noqa: F401  (re-exported)

from . import scenarios as S

BASES = ("thin", "thick", "active", "active_dense")
DEFAULT_SEEDS = 200
DEFAULT_SIM_TIME = 2000
LAM_SLOW = 0.0125


def _group_dict(g) -> dict:
    """WashGroup -> kwargs dict, keeping only fields that differ from the defaults."""
    default = S.WashGroup()
    d = asdict(g)
    out = {k: v for k, v in d.items() if k == "n_wallets" or v != getattr(default, k)}
    if out.get("q_range") is not None:
        out["q_range"] = list(out["q_range"])
    return out


def _evasive(**kw) -> dict:
    kw.setdefault("camouflage", True)
    if kw["camouflage"]:
        kw.setdefault("camouflage_lam", S.CAMO_TOTAL / S.POOL)
    return _group_dict(S.evasive_group(**kw))


def _cells_adversary():
    for bg in BASES:
        spec = dict(S.BACKGROUNDS[bg])
        yield (f"{bg}__control", bg, spec, [],
               dict(background=bg, base=bg, honest_size=1, scenario="control"))
        for gap, order, pipe, lam in product((0, 5), ("sequential", "interleaved"), (1, 4, 8),
                                             (S.LAM_WASH, LAM_SLOW)):
            cell = f"{bg}__gap{gap}_{order[:5]}_pipe{pipe}_lam{_fmt(lam)}"
            yield (cell, bg, spec,
                   [_evasive(gap=gap, edge_order=order, pipeline=pipe, lam_wash=lam)],
                   dict(background=bg, base=bg, honest_size=1, scenario="wash", gap=gap,
                        edge_order=order, pipeline=pipe, lam_wash=lam))


HONEST_ADVERSARIES = {
    "control":       lambda: [],
    "triad":         lambda: [_group_dict(S._g(3))],
    "evasive":       lambda: [_evasive()],
    "evasive_block": lambda: [_evasive(size_exec="block")],
    "evasive_ip8":   lambda: [_evasive(edge_order="interleaved", pipeline=8)],
}


def _cells_honest():
    for base, m in product(BASES, (1, S.HONEST_SIZE)):
        bg = base if m == 1 else f"{base}_mu{m}"
        spec = dict(S.BACKGROUNDS[bg])
        for adv, groups in HONEST_ADVERSARIES.items():
            yield (f"{bg}__{adv}", bg, spec, groups(),
                   dict(background=bg, base=base, honest_size=m, scenario=adv))


SWEEPS = {
    "adversary": (_cells_adversary, "tape"),
    "honest":    (_cells_honest, "tape"),
}
SWEEP_CELLS = {"adversary": 100, "honest": 40}


def enumerate_runs(sweep, seeds=DEFAULT_SEEDS, sim_time=DEFAULT_SIM_TIME, mode=None) -> list:
    """All RunSpecs of a sweep, cell-major then seed (deterministic order)."""
    if sweep not in SWEEPS:
        raise KeyError(f"unknown sweep {sweep!r}; known: {list(SWEEPS)}")
    cells_fn, default_mode = SWEEPS[sweep]
    seed_list = list(range(seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]
    runs = []
    for cell, bg_name, bg_spec, groups, factors in cells_fn():
        for seed in seed_list:
            runs.append(RunSpec(sweep=sweep, cell=cell, seed=seed, background=bg_name,
                                background_spec=bg_spec, groups=groups, factors=factors,
                                sim_time=sim_time, mode=mode or default_mode))
    return runs
