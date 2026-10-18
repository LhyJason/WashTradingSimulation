r"""agents.py -- MultiUnitZI: a ZI trader whose orders can be larger than one unit.

The upstream ``ZIAgent`` always posts ``quantity=1`` (``marketsim/agent/zero_intelligence_agent.py``).
As a result, every honest position is a +-1 random walk, and any trade larger than one unit is
wash on sight. That is why v7 had to split wash rounds into unit trades.
``MultiUnitZI`` lifts that restriction without touching ``marketsim/``.

On each arrival the agent behaves exactly like ZI (same side coin, same valuation offset, same
order-id draw from the global ``random`` module, same eta rule), plus one extra draw:

  * **size** k uniform on {1, ..., m} (``order_size_max`` = m), from the agent's OWN numpy stream
    (seeded per run and agent), so the global ``random`` / torch streams see exactly the draws a
    ZI makes;
  * **price** for k units: the mean of the k marginal private values the order would walk
    through, so a k-unit order is priced at its average unit value:

        BUY : estimate + mean_{i<k} pv(position + i, BUY)  - offset
        SELL: estimate + mean_{i<k} pv(position - i, SELL) + offset

    With k = 1 this is ZI's formula.

The order book already handles quantities above 1 and partial fills (the four-heap splits an order
and keeps its order_id). Positions update per filled piece.

With m = 1 (or before ``configure`` is called) no size is drawn and the agent is a ZI bit for bit
(``run_smoke.py``, check B). Construction makes the same draws as ZI (``PrivateValues``), so
replacing "ZI" by "MZI" in a composition changes nothing else in the run.

Registered as agent type ``"MZI"`` in ``repro.composable.agents.AGENT_REGISTRY`` on import, with the
same builder arguments as ZI.
"""
import random

import numpy as np

from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from repro.composable import agents as _registry

AGENT_TYPE = "MZI"
_SIZE_SEED_OFFSET = 80_021


class MultiUnitZI(ZIAgent):
    order_size_max = 1                       # class default: ZI behaviour until configured
    _size_rng = None

    def configure(self, order_size_max: int, seed=None):
        """Turn on k ~ U{1..order_size_max}; ``seed`` seeds this agent's private size stream."""
        m = int(order_size_max)
        if m < 1:
            raise ValueError(f"order_size_max must be >= 1, got {order_size_max}")
        self.order_size_max = m
        self._size_rng = np.random.default_rng(seed)
        self.sizes = []                      # drawn order sizes, for checks and metrics

    def _draw_size(self) -> int:
        if self.order_size_max <= 1:
            return 1
        k = int(self._size_rng.integers(1, self.order_size_max + 1))
        self.sizes.append(k)
        return k

    def _unit_value(self, side, k) -> float:
        step = 1 if side == BUY else -1
        return float(np.mean([self.pv.value_for_exchange(self.position + step * i, side)
                              for i in range(k)])) if k > 1 else \
            self.pv.value_for_exchange(self.position, side)

    def take_action(self):
        # ZI.take_action re-stated; the only changes are k and the k-unit valuation
        side = random.choice([BUY, SELL])
        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread * random.random() + self.shade[0]
        k = self._draw_size()
        value = self._unit_value(side, k)
        if side == BUY:
            price = estimate + value - valuation_offset
        else:
            price = estimate + value + valuation_offset

        if self.eta != 1.0:
            if side == BUY:
                best_price = self.market.order_book.get_best_ask()
                base_price = estimate + value
                if (base_price - best_price) > self.eta * valuation_offset and best_price != np.inf:
                    price = best_price
            else:
                best_price = self.market.order_book.get_best_bid()
                base_price = estimate + value
                if (best_price - base_price) > self.eta * valuation_offset and best_price != np.inf:
                    price = best_price

        order = Order(
            price=price,
            quantity=k,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=random.randint(1, 10000000)
        )
        return [order]

    def __str__(self):
        return f'MZI{self.agent_id}'


def _build_mzi(agent_id, market, common, params):
    return MultiUnitZI(
        agent_id=agent_id,
        market=market,
        q_max=params.get("q_max", common["q_max"]),
        shade=params.get("shade", common["shade"]),
        pv_var=params.get("pv_var", common["pv_var"]),
        eta=params.get("eta", common["eta"]),
    )


_registry.AGENT_REGISTRY.setdefault(AGENT_TYPE, (_build_mzi, False))


def size_seed(run_seed, agent_id):
    """Seed of an agent's private size stream (None -> unseeded)."""
    return None if run_seed is None else int(run_seed) + _SIZE_SEED_OFFSET + 7 * int(agent_id)
