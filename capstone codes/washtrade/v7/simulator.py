r"""simulator.py -- WashTradingSimulator (v7): v6 + an evasive wash adversary.

v7 changes only the wash controller (see controller.py): a group's ``n_wallets`` is now a wallet
POOL, and each round of the group trades a closed cycle over ``cycle_len`` wallets drawn from it,
with a random size and an optional residual. Everything else -- backgrounds, camouflage (applies
to the whole pool), price rule, cancellation, scheduling, clearing, LOB recorder -- is v6's.

Construction: v6's constructor builds the wallets and one controller per group; v7 then
replaces each controller with the v7 class for the group's price rule, built with exactly the
same arguments plus the v7 knobs. A controller only draws from its own seeded numpy streams, so
building it twice changes nothing else in the run.
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from washtrade.v6.simulator import WashGroup as V6WashGroup
from washtrade.v6.simulator import WashTradingSimulator as V6WashTradingSimulator

from .controller import PRICE_RULES


@dataclass
class WashGroup(V6WashGroup):
    """One wash group. ``n_wallets`` is the wallet pool; the v7 fields are off by default."""
    cycle_len: Optional[int] = None            # wallets per round; None -> the whole pool
    rotation: str = "fixed"                    # "fixed" (v3), "random", "lru"
    q_range: Optional[Tuple[int, int]] = None  # round size Q uniform on {lo..hi}; None -> 1
    size_exec: str = "units"                   # "units" (Q unit events) or "block" (one qty-Q event)
    unit_lam: Optional[float] = None           # wait rate between pieces of one edge; None -> lam_wash
    residual_prob: float = 0.0                 # P(closing edge is off by delta)
    residual_max: int = 1                      # |delta| <= d and |intended inventory| <= d

    def __post_init__(self):
        if self.q_range is not None:
            self.q_range = tuple(int(v) for v in self.q_range)

    def v7_kwargs(self) -> dict:
        return dict(cycle_len=self.cycle_len, rotation=self.rotation, q_range=self.q_range,
                    size_exec=self.size_exec, unit_lam=self.unit_lam,
                    residual_prob=self.residual_prob, residual_max=self.residual_max)


class WashTradingSimulator(V6WashTradingSimulator):

    def __init__(self, agent_composition, *, wash_groups=None, wash_seed=None, **kwargs):
        groups = [g if isinstance(g, WashGroup) else WashGroup(**g) for g in (wash_groups or [])]
        super().__init__(agent_composition, wash_groups=groups, wash_seed=wash_seed, **kwargs)

        market = self.markets[0]
        for k, (g, old) in enumerate(zip(self.groups, list(self.controllers))):
            seed = g.seed if g.seed is not None else (
                None if wash_seed is None else int(wash_seed) + 1000 * k)
            self.controllers[k] = PRICE_RULES[g.price_rule](
                wallets=old.wallets, market=market, q=g.q, eps=g.eps, lam_wash=old.lam_wash,
                gap=g.gap, first_leg=g.first_leg, seed=seed, sim_time=self.sim_time,
                group_id=k, **g.price, **g.v7_kwargs())
        self.run_config["wash_groups"] = [asdict(g) for g in self.groups]
