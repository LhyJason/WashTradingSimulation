r"""wash_agent.py -- WashTrader: a single wash-trading wallet. (identical to v1)

One WashTrader == one wallet == one agent_id == one net position. Several wallets share a
WashController (see controller.py) that decides when and how they trade with each other.

The wallet never acts on its own: unlike ZI/HBL agents it is NOT scheduled into the
simulator's geometric arrival stream. Its orders are built and injected by the controller,
and its position/cash are booked by the upstream matched-order loop (the wallet is a real
agent in ``sim.agents``, so ``update_position`` gets called exactly like for any other agent).

As a pure *volume* manipulator it derives no private consumption value from holding the asset,
so ``get_pos_value()`` is 0 (its end wealth is just cash + position * final_fundamental, and
since it targets net-zero inventory that term is ~0 too).

Copied rather than imported from v1 so that v2 stays a self-contained, independently
runnable snapshot of the mechanism.
"""
from typing import List

from marketsim.agent.agent import Agent
from marketsim.fourheap.order import Order


class WashTrader(Agent):
    def __init__(self, agent_id: int, market):
        self.agent_id = agent_id
        self.market = market
        self.position = 0
        self.cash = 0

    def get_id(self) -> int:
        return self.agent_id

    def take_action(self, *args, **kwargs) -> List[Order]:
        # Driven exclusively by the WashController; never invoked via geometric arrival.
        # Accepts *args so it is safe even if some code path calls take_action(side).
        return []

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def get_pos_value(self) -> float:
        # Pure volume manipulator: no private value for the asset.
        return 0.0

    def reset(self):
        self.position = 0
        self.cash = 0

    def __str__(self):
        return f"WT{self.agent_id}"
