r"""design.py -- the v6 experimental designs (sweeps), as flat lists of run specifications.

A sweep is a full factorial over a few factors, times seeds. Each run is a ``RunSpec``: enough
to rebuild it from scratch (background spec, wash groups, seed, horizon, export mode), plus the
factor values that ``generate.py`` writes to the manifest. The list order is deterministic, so
shard i of n (``index % n == i``) is the same on every machine.

    sweep        factors (x seeds)                                                     runs @20 seeds
    scenarios    5 backgrounds x 13 named scenarios                                          1300
    bg_grid      n_zi {13,30,60,100} x lam {5e-3,2e-2,5e-2} x r {1e-2,3e-3,1e-3}
                 x {control, triad}                                                          1440
    gap          3 backgrounds x wallets {2,3} x gap {0,1,2,3,5,8,12,20,30,50}               1200
    camo         3 backgrounds x gap {0,5} x (off + camo_lam {.002,.005,.01,.02,.05,.1}
                 x n_camouflage {1,3})                                                       1560
    intensity    3 backgrounds x gap {0,5} x lam_wash {.005,.01,.02,.05,.1}                   600
    price        3 backgrounds x gap {0,5,20} x price rule {mid, microprice}                  360
                                                                                     total   6460

The "3 backgrounds" of the frontier sweeps are thin, thick and active (v3's two + the active
honest market). ``scenarios`` is the labelled main dataset (export "full"); the frontier sweeps
default to "slim" exports (no lob_levels.csv) and ``bg_grid`` to "tape" (no LOB at all).
"""
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Optional

from . import scenarios as S

FRONTIER_BACKGROUNDS = ("thin", "thick", "active")
DEFAULT_SEEDS = 20
DEFAULT_SIM_TIME = 2000


@dataclass
class RunSpec:
    sweep: str
    cell: str                       # factor combination, unique within the sweep
    seed: int
    background: str                 # background name (``custom`` specs get a descriptive name)
    background_spec: dict           # composition, lam, r, ...
    groups: list                    # list of WashGroup kwargs dicts
    factors: dict = field(default_factory=dict)
    sim_time: int = DEFAULT_SIM_TIME
    mode: str = "slim"

    @property
    def run_id(self) -> str:
        return f"{self.cell}__s{int(self.seed):03d}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id
        return d


def _fmt(x) -> str:
    """Compact, filename-safe number: 0.005 -> 5e-3, 0.02 -> 2e-2, 20 -> 20."""
    if isinstance(x, float) and x != int(x):
        m, e = f"{x:.0e}".split("e")
        return f"{m}e{int(e)}"
    return str(int(x)) if isinstance(x, float) else str(x)


def _group(n_wallets=3, **kw) -> dict:
    kw.setdefault("lam_wash", S.LAM_WASH)
    return dict(n_wallets=n_wallets, **kw)


def _named_bg(name):
    return name, dict(S.BACKGROUNDS[name])


# --------------------------------------------------------------------------- #
# sweeps: each yields (cell, background name, background spec, groups, factors)
# --------------------------------------------------------------------------- #
def _cells_scenarios():
    for bg, sc in product(S.BACKGROUNDS, S.SCENARIOS):
        name, spec = _named_bg(bg)
        groups = [_asdict_group(g) for g in S.SCENARIOS[sc]()]
        yield f"{bg}__{sc}", name, spec, groups, dict(background=bg, scenario=sc)


def _cells_bg_grid():
    for n, lam, r, sc in product((13, 30, 60, 100), (5e-3, 2e-2, 5e-2), (1e-2, 3e-3, 1e-3),
                                 ("control", "triad")):
        name = f"zi{n}_lam{_fmt(lam)}_r{_fmt(r)}"
        spec = S.background_spec(n, lam, r)
        groups = [] if sc == "control" else [_group(3)]
        yield f"{name}__{sc}", name, spec, groups, dict(n_zi=n, lam=lam, r=r, scenario=sc)


def _cells_gap():
    for bg, nw, gap in product(FRONTIER_BACKGROUNDS, (2, 3), (0, 1, 2, 3, 5, 8, 12, 20, 30, 50)):
        name, spec = _named_bg(bg)
        yield (f"{bg}__w{nw}_gap{gap}", name, spec, [_group(nw, gap=gap)],
               dict(background=bg, n_wallets=nw, gap=gap))


def _cells_camo():
    rates = (2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1)
    for bg, gap in product(FRONTIER_BACKGROUNDS, (0, 5)):
        name, spec = _named_bg(bg)
        yield (f"{bg}__gap{gap}_camo-off", name, spec, [_group(3, gap=gap)],
               dict(background=bg, gap=gap, camouflage_lam=0.0, n_camouflage=0))
        for lam_c, n_c in product(rates, (1, 3)):
            g = _group(3, gap=gap, camouflage=True, camouflage_lam=lam_c, n_camouflage=n_c)
            yield (f"{bg}__gap{gap}_camo{_fmt(lam_c)}_n{n_c}", name, spec, [g],
                   dict(background=bg, gap=gap, camouflage_lam=lam_c, n_camouflage=n_c))


def _cells_intensity():
    for bg, gap, lw in product(FRONTIER_BACKGROUNDS, (0, 5), (5e-3, 1e-2, 2e-2, 5e-2, 1e-1)):
        name, spec = _named_bg(bg)
        yield (f"{bg}__gap{gap}_lw{_fmt(lw)}", name, spec, [_group(3, gap=gap, lam_wash=lw)],
               dict(background=bg, gap=gap, lam_wash=lw))


def _cells_price():
    for bg, gap, rule in product(FRONTIER_BACKGROUNDS, (0, 5, 20), ("mid", "microprice")):
        name, spec = _named_bg(bg)
        yield (f"{bg}__gap{gap}_{rule}", name, spec, [_group(3, gap=gap, price_rule=rule)],
               dict(background=bg, gap=gap, price_rule=rule))


def _asdict_group(g) -> dict:
    """WashGroup -> kwargs dict, keeping only fields that differ from the defaults."""
    default = S.WashGroup()
    d = asdict(g)
    return {k: v for k, v in d.items() if k == "n_wallets" or v != getattr(default, k)}


SWEEPS = {
    "scenarios": (_cells_scenarios, "full"),
    "bg_grid":   (_cells_bg_grid, "tape"),
    "gap":       (_cells_gap, "slim"),
    "camo":      (_cells_camo, "slim"),
    "intensity": (_cells_intensity, "slim"),
    "price":     (_cells_price, "slim"),
}


def enumerate_runs(sweep, seeds=DEFAULT_SEEDS, sim_time=DEFAULT_SIM_TIME,
                   mode: Optional[str] = None) -> list:
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


def shard(runs, index, n_shards) -> list:
    """Runs assigned to shard ``index`` of ``n_shards`` (round robin, so shards are balanced
    across cells of different cost)."""
    if not 0 <= index < n_shards:
        raise ValueError(f"shard index {index} not in [0, {n_shards})")
    return [r for i, r in enumerate(runs) if i % n_shards == index]
