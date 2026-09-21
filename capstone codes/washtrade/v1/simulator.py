r"""simulator.py -- WashTradingSimulator: ComposableSimulator + a wash-trading layer.

It reuses the entire upstream matching / arrival / settlement machinery and adds exactly two
things:

  1) __init__ : accepts a composition that may include "WashTrader": n. The n wash wallets are
     built as real agents in ``self.agents`` (so positions are booked by the upstream
     matched-order loop) but are NOT scheduled into ``self.arrivals`` -- they are driven only
     by the controller. All other agent types are built by ComposableSimulator as usual.

  2) step()/run() : before a timestep's clear, if a wash event is due, the controller's
     crossing (SELL, BUY) pair is injected into the SAME timestep, so it participates in the
     one batch clear that ComposableSimulator already performs. run() is widened to also fire
     on wash-due timesteps (which may have no ZI arrival of their own).

Nothing in marketsim/ or repro/composable/ is modified.
"""
from repro.composable.simulator import ComposableSimulator
from repro.composable.composition import normalize_composition

from .wash_agent import WashTrader
from .controller import WashController


class WashTradingSimulator(ComposableSimulator):
    WASH_TYPE = "WashTrader"

    def __init__(self, agent_composition, *, wash_q=1, wash_eps=0.0,
                 lam_wash=None, wash_seed=None, n_wallets=None, **kwargs):
        specs = normalize_composition(agent_composition)
        wash_count = sum(s.count for s in specs if s.agent_type == self.WASH_TYPE)
        background = [s for s in specs if s.agent_type != self.WASH_TYPE]

        if n_wallets is None:
            n_wallets = wash_count            # all-ZI control -> 0 wallets -> controller idle

        # Build background agents (ZI/HBL/...) exactly as upstream: consecutive ids,
        # geometric arrivals scheduled. Wash wallets are intentionally excluded here.
        super().__init__(background, **kwargs)

        # Append the wash wallets AFTER the background, with fresh consecutive ids. They are
        # added to self.agents/self.agent_types but NOT to self.arrivals.
        market = self.markets[0]
        start_id = len(self.agents)
        self.wash_wallet_ids = []
        for k in range(n_wallets):
            aid = start_id + k
            self.agents[aid] = WashTrader(aid, market)
            self.agent_types[aid] = self.WASH_TYPE
            self.wash_wallet_ids.append(aid)
        self.num_agents += n_wallets

        if lam_wash is None:
            lam_wash = self.lam
        self.controller = WashController(
            wallets=[self.agents[i] for i in self.wash_wallet_ids],
            market=market, q=wash_q, eps=wash_eps, lam_wash=lam_wash,
            seed=wash_seed, sim_time=self.sim_time,
        )

    def step(self):
        if self.time < self.sim_time and self.controller.due(self.time):
            market = self.markets[0]
            market.event_queue.set_time(self.time)   # so the fundamental anchor reads at self.time
            seller_id, buyer_id, sell_order, buy_order = self.controller.make_wash_pair(self.time)
            market.withdraw_all(seller_id)            # clear any stale resting wash orders
            market.withdraw_all(buyer_id)
            market.add_orders([sell_order, buy_order])
            n_before = len(market.matched_orders)
            super().step()                            # processes ZI arrivals + single batch clear
            new_matched = market.matched_orders[n_before:]
            self.controller.record(self.time, new_matched)
            self.controller.advance()
        else:
            super().step()

    def run(self):
        # Same skeleton as upstream run(), but also fire on wash-due timesteps (which may have
        # no background arrival). self.time stays in lock-step with the loop index t.
        for t in range(self.sim_time):
            if self.arrivals[t] or self.controller.due(t):
                self.step()
            self.time += 1
        self.step()
