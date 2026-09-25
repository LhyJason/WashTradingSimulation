"""Reuse v3 mechanics with microprice-centered pricing controllers."""

from dataclasses import dataclass
from typing import Optional

from washtrade.v3.simulator import WashGroup as V3WashGroup
from washtrade.v3.simulator import WashTradingSimulator as V3WashTradingSimulator

from .controller import WashController


@dataclass
class WashGroup(V3WashGroup):
    price_range_fraction: float = 0.8
    price_sigma_fraction: float = 0.5
    volatility_multiplier: float = 0.5
    volatility_lookback: int = 20
    max_abs_range: float = 250.0
    fallback_range: float = 100.0
    tick_size: float = 1.0
    use_microprice: bool = True


class WashTradingSimulator(V3WashTradingSimulator):
    def __init__(self, agent_composition, *, wash_groups=None, **kwargs):
        if wash_groups is None:
            raise ValueError("v5 requires explicit wash_groups")
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
                volatility_multiplier=group.volatility_multiplier,
                volatility_lookback=group.volatility_lookback,
                max_abs_range=group.max_abs_range,
                fallback_range=group.fallback_range, tick_size=group.tick_size,
                use_microprice=group.use_microprice,
            ))
