r"""simulator.py -- WashTradingSimulator (v2): ComposableSimulator + a wash-trading layer.

It reuses the entire upstream matching / arrival / settlement machinery and adds exactly
three things:

  1) __init__ : accepts a composition that may include "WashTrader": n. The n wash wallets are
     built as real agents in ``self.agents`` (so positions are booked by the upstream
     matched-order loop) but are NOT scheduled into ``self.arrivals`` -- they are driven only
     by the controller. n = 2 (ping-pong) and n = 3 (triangle) are both supported.

  2) step()/run() : if a wash LEG is due at this timestep, the controller's order for that leg
     is injected before the timestep's single batch clear. With ``wash_gap=0`` both legs are
     injected together (v1 behaviour); with ``wash_gap>0`` leg 2 arrives ``gap`` timesteps
     later, and every background arrival in between can hit the exposed leg 1. run() fires on
     any timestep that carries a background arrival OR a wash leg.

  3) LOB recording : the book is snapshotted around every cleared timestep (see lob.py).
     Set ``record_lob=False`` to switch it off.

Nothing in marketsim/ or repro/composable/ is modified.
"""
from repro.composable.simulator import ComposableSimulator
from repro.composable.composition import normalize_composition

from .wash_agent import WashTrader
from .controller import WashController
from .lob import LOBRecorder


class WashTradingSimulator(ComposableSimulator):
    WASH_TYPE = "WashTrader"

    def __init__(self, agent_composition, *, wash_q=1, wash_eps=0.0, wash_gap=0,
                 wash_first_leg="random", lam_wash=None, wash_seed=None, n_wallets=None,
                 record_lob=True, lob_depth_levels=5, **kwargs):
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
            market=market, q=wash_q, eps=wash_eps, lam_wash=lam_wash, gap=wash_gap,
            first_leg=wash_first_leg, seed=wash_seed, sim_time=self.sim_time,
        )

        self.lob = LOBRecorder(wash_ids=self.wash_wallet_ids,
                               depth_levels=lob_depth_levels, enabled=record_lob)

        # kept for reproducibility / meta.json export
        self.run_config = dict(kwargs)        # sim_time, lam, shade, ... as passed by the caller
        self.run_config.update(
            composition=(dict(agent_composition) if isinstance(agent_composition, dict)
                         else [(s.agent_type, s.count) for s in specs]),
            n_wallets=n_wallets, wash_q=wash_q, wash_eps=wash_eps, wash_gap=wash_gap,
            wash_first_leg=wash_first_leg, lam_wash=lam_wash, wash_seed=wash_seed,
            record_lob=record_lob, lob_depth_levels=lob_depth_levels,
            sim_time=self.sim_time, lam=self.lam, num_agents=self.num_agents,
            wash_wallet_ids=list(self.wash_wallet_ids),
        )

    def step(self):
        market = self.markets[0]
        t = self.time
        pre = self.lob.book_state(market) if self.lob.enabled else None

        posts = []
        if t < self.sim_time and self.controller.due(t):
            market.event_queue.set_time(t)    # so leg 1 reads the fundamental/quotes at t
            posts = self.controller.orders_for(t)
            for aid, order in posts:
                # only the posting wallet's own stale orders are pulled -- an exposed leg 1
                # belonging to the OTHER wallet must survive until its partner arrives.
                market.withdraw_all(aid)
                market.add_orders([order])

        n_before = len(market.matched_orders)
        super().step()                        # ZI arrivals + the single batch clear
        new_matched = market.matched_orders[n_before:]

        if t < self.sim_time:
            self.controller.observe(t, new_matched)
            if self.lob.enabled:
                self.lob.record(market, t, pre, new_matched,
                                posts=posts, controller=self.controller)

    def run(self):
        # Same skeleton as upstream run(), but also fire on wash-leg timesteps (which may have
        # no background arrival). self.time stays in lock-step with the loop index t.
        for t in range(self.sim_time):
            if self.arrivals[t] or self.controller.due(t):
                self.step()
            self.time += 1
        self.step()
        self.controller.finalize()            # close out a leg-2 that ran off the end
