r"""design.py -- the v7 experimental designs (sweeps), as flat lists of run specifications.

Same machinery as v6 (``RunSpec``, deterministic order, round-robin shards). Default 50 seeds.

    sweep       factors (x seeds)                                                   cells  runs @50
    scenarios   5 backgrounds x 12 named scenarios (4 v6 baselines + 8 v7)            60     3000
    rotation    3 bg x cycle {2,3} x gap {0,5} x pool {k (fixed),6,12,24,48,96}       72     3600
    size        3 bg x gap {0,5} x (Q=1 + Q~U{1..hi}, hi {2,4,8} x exec {units,block}) 42     2100
    residual    3 bg x gap {0,5} x (off + p {.1,.25,.5,1} x d {1,2})                   54     2700
    evasive     4 bg x gap {0,5,20} x pool {3,12,96} x Q {1, 1..4} x residual {0,.5}
                x camouflage {off, on}  + one control per background                   292    14600
                                                                                total   520    26000

"3 bg" = thin, thick, active; ``evasive`` adds active_dense. Single-factor sweeps vary one v7
knob on the standard triad (lam_wash 0.05, rotation "random" when pool > k); ``evasive`` is the
full factorial of all three knobs with gap and camouflage, and carries the control runs that a
detector's threshold calibration needs (same seeds -> same control runs for every sweep).
Camouflage "on" = the whole pool camouflages with total rate ``S.CAMO_TOTAL`` (0.06), i.e.
0.06 / pool per wallet, so the market gains the same ordinary activity at every pool size.
``scenarios`` is exported "slim" (with lob.csv); every other sweep "tape" (trades + labels).
"""
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Optional

from washtrade.v6.design import _fmt

from . import scenarios as S

FRONTIER_BACKGROUNDS = ("thin", "thick", "active")
EVASIVE_BACKGROUNDS = ("thin", "thick", "active", "active_dense")
SCENARIO_NAMES = ("control", "triad", "dyad", "triad_gap5",
                  "triad_pool12", "triad_pool96", "triad_size4", "triad_block4", "triad_res",
                  "triad_evasive", "triad_evasive_g5", "evasive_camo")
DEFAULT_SEEDS = 50
DEFAULT_SIM_TIME = 2000


@dataclass
class RunSpec:
    sweep: str
    cell: str
    seed: int
    background: str
    background_spec: dict
    groups: list
    factors: dict = field(default_factory=dict)
    sim_time: int = DEFAULT_SIM_TIME
    mode: str = "tape"

    @property
    def run_id(self) -> str:
        return f"{self.cell}__s{int(self.seed):03d}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id
        return d


def _named_bg(name):
    return name, dict(S.BACKGROUNDS[name])


def _group_dict(g) -> dict:
    """WashGroup -> kwargs dict, keeping only fields that differ from the defaults."""
    default = S.WashGroup()
    d = asdict(g)
    out = {k: v for k, v in d.items() if k == "n_wallets" or v != getattr(default, k)}
    if out.get("q_range") is not None:
        out["q_range"] = list(out["q_range"])
    return out


def _triad(pool=3, cycle_len=None, **kw) -> dict:
    k = pool if cycle_len is None else cycle_len
    rotation = "fixed" if pool == k else "random"
    return _group_dict(S._g(pool, cycle_len=(None if pool == k else k), rotation=rotation, **kw))


# --------------------------------------------------------------------------- #
# sweeps: each yields (cell, background name, background spec, groups, factors)
# --------------------------------------------------------------------------- #
def _cells_scenarios():
    for bg, sc in product(S.BACKGROUNDS, SCENARIO_NAMES):
        name, spec = _named_bg(bg)
        groups = [_group_dict(g) for g in S.SCENARIOS[sc]()]
        yield f"{bg}__{sc}", name, spec, groups, dict(background=bg, scenario=sc)


def _cells_rotation():
    for bg, k, gap in product(FRONTIER_BACKGROUNDS, (2, 3), (0, 5)):
        name, spec = _named_bg(bg)
        for pool in (k, 6, 12, 24, 48, 96):
            yield (f"{bg}__k{k}_gap{gap}_pool{pool}", name, spec,
                   [_triad(pool, k, gap=gap)],
                   dict(background=bg, cycle_len=k, gap=gap, pool=pool))


def _cells_size():
    for bg, gap in product(FRONTIER_BACKGROUNDS, (0, 5)):
        name, spec = _named_bg(bg)
        yield (f"{bg}__gap{gap}_q1", name, spec, [_triad(gap=gap)],
               dict(background=bg, gap=gap, q_hi=1, size_exec="units"))
        for hi, ex in product((2, 4, 8), ("units", "block")):
            yield (f"{bg}__gap{gap}_q1-{hi}_{ex}", name, spec,
                   [_triad(gap=gap, q_range=(1, hi), size_exec=ex)],
                   dict(background=bg, gap=gap, q_hi=hi, size_exec=ex))


def _cells_residual():
    for bg, gap in product(FRONTIER_BACKGROUNDS, (0, 5)):
        name, spec = _named_bg(bg)
        yield (f"{bg}__gap{gap}_res-off", name, spec, [_triad(gap=gap)],
               dict(background=bg, gap=gap, residual_prob=0.0, residual_max=0))
        for p, d in product((0.1, 0.25, 0.5, 1.0), (1, 2)):
            yield (f"{bg}__gap{gap}_res{_fmt(p)}_d{d}", name, spec,
                   [_triad(gap=gap, residual_prob=p, residual_max=d)],
                   dict(background=bg, gap=gap, residual_prob=p, residual_max=d))


def _cells_evasive():
    for bg in EVASIVE_BACKGROUNDS:
        name, spec = _named_bg(bg)
        yield (f"{bg}__control", name, spec, [],
               dict(background=bg, scenario="control"))
        for gap, pool, q_hi, res, camo in product((0, 5, 20), (3, 12, 96), (1, 4), (0.0, 0.5),
                                                  (False, True)):
            kw = dict(gap=gap)
            if q_hi > 1:
                kw["q_range"] = (1, q_hi)
            if res > 0:
                kw["residual_prob"] = res
            if camo:
                kw.update(camouflage=True, camouflage_lam=S.CAMO_TOTAL / pool)
            cell = (f"{bg}__gap{gap}_pool{pool}_q1-{q_hi}_res{_fmt(res)}"
                    f"_camo{'on' if camo else 'off'}")
            yield (cell, name, spec, [_triad(pool, 3, **kw)],
                   dict(background=bg, scenario="wash", gap=gap, pool=pool, q_hi=q_hi,
                        residual_prob=res, camouflage_total=(S.CAMO_TOTAL if camo else 0.0)))


SWEEPS = {
    "scenarios": (_cells_scenarios, "slim"),
    "rotation":  (_cells_rotation, "tape"),
    "size":      (_cells_size, "tape"),
    "residual":  (_cells_residual, "tape"),
    "evasive":   (_cells_evasive, "tape"),
}
SWEEP_CELLS = {"scenarios": 60, "rotation": 72, "size": 42, "residual": 54, "evasive": 292}


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
    """Runs assigned to shard ``index`` of ``n_shards`` (round robin)."""
    if not 0 <= index < n_shards:
        raise ValueError(f"shard index {index} not in [0, {n_shards})")
    return [r for i, r in enumerate(runs) if i % n_shards == index]
