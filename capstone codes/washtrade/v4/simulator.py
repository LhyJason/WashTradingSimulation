"""Reuse v3 market mechanics with the randomized-price wash controller."""

from dataclasses import dataclass
from typing import Optional

from washtrade.v3.simulator import WashGroup as V3WashGroup
from washtrade.v3.simulator import WashTradingSimulator as V3WashTradingSimulator

from .controller import WashController


@dataclass
class WashGroup(V3WashGroup):
    # The sell-limit offset is N(0, (radius * price_sigma_fraction)^2),
    # truncated to +/-radius; the buy limit adds a positive random width.
    # With both quotes present, radius is half-spread * price_range_fraction.
    price_range_fraction: float = 0.5
    price_sigma_fraction: float = 0.5
    price_width_fraction: float = 0.25
    # Absolute half-range before price_range_fraction when the book is empty.
    fallback_range: float = 100.0


class WashTradingSimulator(V3WashTradingSimulator):
    def __init__(self, agent_composition, *, wash_groups=None, **kwargs):
        if wash_groups is None:
            raise ValueError("v4 requires explicit wash_groups")
        groups = [g if isinstance(g, WashGroup) else WashGroup(**g) for g in wash_groups]
        super().__init__(agent_composition, wash_groups=groups, **kwargs)
        wash_seed = kwargs.get("wash_seed")
        market = self.markets[0]
        self.controllers = []
        for k, group in enumerate(groups):
            ids = [aid for aid, group_id in self.wallet_group.items() if group_id == k]
            seed: Optional[int] = group.seed if group.seed is not None else (
                None if wash_seed is None else int(wash_seed) + 1000 * k)
            self.controllers.append(WashController(
                wallets=[self.agents[aid] for aid in ids], market=market,
                q=group.q, eps=group.eps,
                lam_wash=group.lam_wash if group.lam_wash is not None else self.lam,
                gap=group.gap, first_leg=group.first_leg, seed=seed,
                sim_time=self.sim_time, group_id=k,
                price_range_fraction=group.price_range_fraction,
                price_sigma_fraction=group.price_sigma_fraction,
                price_width_fraction=group.price_width_fraction,
                fallback_range=group.fallback_range,
            ))
