r"""wash_agent.py -- WashTrader: a pure (non-camouflage) wash-trading wallet. (identical to v2)

One WashTrader == one wallet == one agent_id == one net position. Several wallets share a
WashController (see controller.py) that decides when and how they trade with each other.

The wallet never acts on its own: it is NOT scheduled into the simulator's geometric arrival
stream. Its orders are built and injected by the controller, and its position/cash are booked
by the matched-order loop exactly like any other agent's.

Camouflage wallets do NOT use this class: they are ordinary ZI agents that a controller also
drives (see ``simulator.WashGroup``).
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
