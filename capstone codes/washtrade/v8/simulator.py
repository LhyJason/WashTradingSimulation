r"""simulator.py -- WashTradingSimulator (v8): v7 + execution-order knobs + multi-unit honest orders.

Two changes relative to v7:

1. **Wash execution** (``WashGroup.edge_order``, ``WashGroup.pipeline``; see controller.py). As in
   v7, the v7 controller of each group is replaced by the v8 class for its price rule, built with
   the same arguments plus the v8 knobs.
2. **Multi-unit honest orders** (simulator keyword ``order_size_max`` = m). With m set, every
   ``"ZI"`` in the background composition becomes ``"MZI"`` (``agents.MultiUnitZI``). So does
   every group's ``camouflage_type``, so camouflage wallets keep trading like the background.
   Each MZI then draws its order size uniformly from {1..m}, from a private stream seeded by the
   run seed and its agent id. Construction makes the same draws as ZI, so the run is the same
   up to the order sizes. m = None (default) leaves v7's plain ZI.

With default arguments v8 reproduces v7 bit for bit, and so does m = 1 (MZI without size draws).
See run_smoke.py, checks A and B.
"""
from dataclasses import asdict, dataclass, replace

from washtrade.v7.simulator import WashGroup as V7WashGroup
from washtrade.v7.simulator import WashTradingSimulator as V7WashTradingSimulator

from . import agents as AG
from .controller import PRICE_RULES


@dataclass
class WashGroup(V7WashGroup):
    """One wash group: v7's fields plus the v8 execution knobs (off by default)."""
    edge_order: str = "sequential"            # "sequential" (v7) or "interleaved"
    pipeline: int = 1                         # rounds open at once (1 = v7)

    def v8_kwargs(self) -> dict:
        return dict(edge_order=self.edge_order, pipeline=self.pipeline)


def _multi_unit_composition(comp):
    """{"ZI": n, ...} / list form -> the same with every ZI as MZI."""
    if isinstance(comp, dict):
        return {(AG.AGENT_TYPE if t == "ZI" else t): n for t, n in comp.items()}
    out = []
    for item in comp:
        if isinstance(item, (tuple, list)) and item and item[0] == "ZI":
            out.append((AG.AGENT_TYPE,) + tuple(item[1:]))
        elif hasattr(item, "agent_type") and item.agent_type == "ZI":
            out.append(replace(item, agent_type=AG.AGENT_TYPE))
        else:
            out.append(item)
    return out


class WashTradingSimulator(V7WashTradingSimulator):

    def __init__(self, agent_composition, *, wash_groups=None, wash_seed=None,
                 order_size_max=None, **kwargs):
        groups = [g if isinstance(g, WashGroup) else WashGroup(**g) for g in (wash_groups or [])]
        if order_size_max is not None:
            agent_composition = _multi_unit_composition(agent_composition)
            groups = [replace(g, camouflage_type=AG.AGENT_TYPE) if g.camouflage_type == "ZI" else g
                      for g in groups]
        super().__init__(agent_composition, wash_groups=groups, wash_seed=wash_seed, **kwargs)

        market = self.markets[0]
        for k, (g, old) in enumerate(zip(self.groups, list(self.controllers))):
            seed = g.seed if g.seed is not None else (
                None if wash_seed is None else int(wash_seed) + 1000 * k)
            self.controllers[k] = PRICE_RULES[g.price_rule](
                wallets=old.wallets, market=market, q=g.q, eps=g.eps, lam_wash=old.lam_wash,
                gap=g.gap, first_leg=g.first_leg, seed=seed, sim_time=self.sim_time,
                group_id=k, **g.price, **g.v7_kwargs(), **g.v8_kwargs())

        self.order_size_max = order_size_max
        if order_size_max is not None:
            for aid, agent in self.agents.items():
                real = getattr(agent, "_agent", agent)          # unwrap the action adapter
                if isinstance(real, AG.MultiUnitZI):
                    real.configure(order_size_max, AG.size_seed(wash_seed, aid))
        self.run_config["wash_groups"] = [asdict(g) for g in self.groups]
        self.run_config["order_size_max"] = order_size_max

    def honest_order_sizes(self) -> list:
        """Every order size drawn by a multi-unit agent (camouflage wallets included)."""
        out = []
        for agent in self.agents.values():
            real = getattr(agent, "_agent", agent)
            out.extend(getattr(real, "sizes", []))
        return out
