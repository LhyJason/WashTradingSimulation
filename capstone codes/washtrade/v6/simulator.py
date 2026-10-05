r"""simulator.py -- WashTradingSimulator (v6): v3 mechanics + background knobs + camouflage intensity.

What v6 changes relative to v3 (v3 itself is untouched):

1. **Camouflage intensity** (``WashGroup.camouflage_lam``). In v3 a camouflage wallet arrives
   at the background rate and draws its arrival times from the background's shared torch stream.
   v6 lets each group give its camouflage wallets their OWN arrival rate. Such a wallet then
   draws its inter-arrival times from a private numpy stream (one per wallet), so it never
   consumes the shared stream: the background agents' arrival SCHEDULE is the same whatever the
   camouflage rate (and the same as in a run without camouflage). Their actions still differ,
   because every ZI action draws from the global ``random`` module and the book they see changes.
   ``camouflage_lam=None`` keeps v3's shared-stream behaviour exactly.

2. **Partial camouflage** (``WashGroup.n_camouflage``). Only the first ``n_camouflage`` wallets
   of the group also trade as background agents; the rest are pure wash wallets. ``None`` means
   all wallets when ``camouflage=True`` (v3) and none otherwise.

3. **Price rule per group** (``WashGroup.price_rule``): ``"mid"`` = v3 midpoint legs (default),
   ``"microprice"`` = the v5 controller (one random common price per event, centred on the
   microprice; parameters in ``WashGroup.price``).

4. **Background knobs.** Every ``ComposableSimulator`` keyword (``r``, ``shock_var``, ``pv_var``,
   ``shade``, ``q_max``, ...) passes straight through, as in v3. The one that matters is the mean
   reversion ``r``: ZIs value the asset at their forecast of the FINAL fundamental,

       estimate_t = (1 - rho_t) * mean + rho_t * f_t,   rho_t = (1 - r) ** (T - t),

   so with the Table-3 value r = 0.01 and T = 2000, rho_t is ~0 for most of the run, every ZI
   values the asset at the constant ``mean``, and honest trading stops after an opening burst
   and only resumes in the last ~300 steps. r = 1e-3 keeps the honest market active throughout.
   See ``scenarios.BACKGROUNDS``.

With default arguments (and ``price_rule="mid"``) v6 reproduces v3 bit for bit; with
``price_rule="microprice"`` it reproduces v5 bit for bit (``run_smoke.py`` checks A and B).
"""
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from marketsim.simulator.sampled_arrival_simulator import sample_arrivals
from repro.composable.agents import make_agent
from repro.composable.composition import normalize_composition
from repro.composable.simulator import ComposableSimulator

from washtrade.v3.controller import WashController as MidController
from washtrade.v3.lob import LOBRecorder
from washtrade.v3.simulator import WashGroup as V3WashGroup
from washtrade.v3.simulator import WashTradingSimulator as V3WashTradingSimulator
from washtrade.v3.simulator import _composable_defaults
from washtrade.v3.wash_agent import WashTrader
from washtrade.v5.controller import WashController as MicropriceController

PRICE_RULES = {"mid": MidController, "microprice": MicropriceController}

# seed offset of the per-wallet camouflage arrival streams (wallet j of a group with seed s
# uses s + _CAMO_ARRIVAL_OFFSET + j)
_CAMO_ARRIVAL_OFFSET = 50_021


@dataclass
class WashGroup(V3WashGroup):
    """One colluding wash group. Fields beyond v3's are all off by default."""
    camouflage_lam: Optional[float] = None   # own arrival rate of camouflage wallets; None -> background lam, shared stream
    n_camouflage: Optional[int] = None       # how many wallets camouflage; None -> all if camouflage else 0
    price_rule: str = "mid"                  # "mid" (v3) or "microprice" (v5)
    price: dict = field(default_factory=dict)  # extra kwargs for the microprice controller

    def camouflage_count(self) -> int:
        if self.n_camouflage is None:
            return self.n_wallets if self.camouflage else 0
        return int(self.n_camouflage)


class _OwnArrivals:
    """Geometric arrivals at rate ``lam`` from a private numpy stream.

    Same law as the upstream torch sampler: the first arrival is at Geom0 (failures before the
    first success, support 0, 1, ...) and each later one at t + 1 + Geom0."""

    def __init__(self, lam, seed):
        if not 0 < lam <= 1:
            raise ValueError(f"camouflage_lam must be in (0, 1], got {lam}")
        self.lam = float(lam)
        self.rng = np.random.default_rng(seed)

    def first(self) -> int:
        return int(self.rng.geometric(self.lam)) - 1

    def after(self, t) -> int:
        return int(t) + int(self.rng.geometric(self.lam))


class WashTradingSimulator(V3WashTradingSimulator):
    """v3 simulator with per-group camouflage intensity, partial camouflage and price rule.

    ``__init__`` re-states v3's constructor (same construction order, so the default path draws
    exactly the same random numbers); ``step`` re-states v3's step and only changes how an
    own-rate camouflage wallet is rescheduled. Cancellation, scheduling of wash legs, clearing,
    booking and the LOB recorder are v3's.
    """

    def __init__(self, agent_composition, *, wash_groups=None, wash_seed=None,
                 record_lob=True, lob_depth_levels=5, **kwargs):
        specs = normalize_composition(agent_composition)
        if any(s.agent_type == self.WASH_TYPE for s in specs):
            raise ValueError("v6 takes wash wallets only via wash_groups")
        groups = [g if isinstance(g, WashGroup) else WashGroup(**g) for g in (wash_groups or [])]

        ComposableSimulator.__init__(self, specs, **kwargs)
        market = self.markets[0]

        defaults = _composable_defaults()
        common = {k: kwargs.get(k, defaults[k]) for k in ("q_max", "pv_var", "shade", "eta", "L")}
        if common["shade"] is None:
            common["shade"] = [250, 500]
        common["lam"] = self.lam

        self.groups = groups
        self.controllers = []
        self.wash_wallet_ids = []
        self.camouflage_ids = []
        self.wallet_group = {}
        self.own_arrivals = {}                        # agent_id -> _OwnArrivals
        next_id = len(self.agents)
        for k, g in enumerate(groups):
            if g.n_wallets < 2:
                raise ValueError(f"wash group {k} needs >= 2 wallets, got {g.n_wallets}")
            n_camo = g.camouflage_count()
            if not 0 <= n_camo <= g.n_wallets:
                raise ValueError(f"wash group {k}: n_camouflage={n_camo} not in [0, {g.n_wallets}]")
            if g.camouflage_lam is not None and n_camo == 0:
                raise ValueError(f"wash group {k}: camouflage_lam set but no camouflage wallets")
            if g.price_rule not in PRICE_RULES:
                raise ValueError(f"wash group {k}: price_rule must be one of {list(PRICE_RULES)}")
            if g.price and g.price_rule != "microprice":
                raise ValueError(f"wash group {k}: price parameters need price_rule='microprice'")

            seed = g.seed if g.seed is not None else (
                None if wash_seed is None else int(wash_seed) + 1000 * k)
            ids = []
            for j in range(g.n_wallets):
                aid = next_id
                next_id += 1
                if j < n_camo:
                    self.agents[aid] = make_agent(g.camouflage_type, aid, market, common, {})
                    if g.camouflage_lam is None:          # v3: shared background stream
                        if self.arrival_index == self.arrivals_sampled:
                            self.arrival_times = sample_arrivals(self.lam_r, self.arrivals_sampled)
                            self.arrival_index = 0
                        self.arrivals[self.arrival_times[self.arrival_index].item()].append(aid)
                        self.arrival_index += 1
                    else:                                  # own rate, private stream
                        own = _OwnArrivals(g.camouflage_lam, None if seed is None
                                           else int(seed) + _CAMO_ARRIVAL_OFFSET + j)
                        self.own_arrivals[aid] = own
                        self.arrivals[own.first()].append(aid)
                    self.camouflage_ids.append(aid)
                else:
                    self.agents[aid] = WashTrader(aid, market)
                self.agent_types[aid] = self.WASH_TYPE
                self.wallet_group[aid] = k
                ids.append(aid)
            self.wash_wallet_ids.extend(ids)

            ctrl_kw = dict(
                wallets=[self.agents[i] for i in ids], market=market,
                q=g.q, eps=g.eps, lam_wash=(g.lam_wash if g.lam_wash is not None else self.lam),
                gap=g.gap, first_leg=g.first_leg, seed=seed, sim_time=self.sim_time, group_id=k)
            self.controllers.append(PRICE_RULES[g.price_rule](**ctrl_kw, **g.price))
        self.num_agents = len(self.agents)

        self.lob = LOBRecorder(depth_levels=lob_depth_levels, enabled=record_lob)

        self.run_config = dict(kwargs)
        self.run_config.update(
            composition=[(s.agent_type, s.count) for s in specs],
            wash_groups=[asdict(g) for g in groups], wash_seed=wash_seed,
            record_lob=record_lob, lob_depth_levels=lob_depth_levels,
            sim_time=self.sim_time, lam=self.lam, num_agents=self.num_agents,
            r=float(kwargs.get("r", defaults["r"])),   # as given (the fundamental stores float32)
            wash_wallet_ids=list(self.wash_wallet_ids),
            camouflage_ids=list(self.camouflage_ids),
            own_rate_camouflage_ids=sorted(self.own_arrivals),
        )

    # -- simulation loop ----------------------------------------------------- #
    def step(self):
        """v3's step; an own-rate camouflage wallet is rescheduled from its private stream."""
        t = self.time
        if t >= self.sim_time:
            return self.end_sim()
        market = self.markets[0]
        pre = self.lob.book_state(market) if self.lob.enabled else None
        market.event_queue.set_time(t)

        posts = []
        for ctrl in self._posting_controllers(t):
            for aid, order in ctrl.orders_for(t):
                self._cancel(aid, wash=True)
                market.add_orders([order])
                posts.append((ctrl.group_id, aid, order))

        for agent_id in self.arrivals[t]:
            agent = self.agents[agent_id]
            self._cancel(agent_id, wash=False)
            orders = agent.take_action()
            market.add_orders(orders)
            own = self.own_arrivals.get(agent_id)
            if own is not None:
                self.arrivals[own.after(t)].append(agent_id)
                continue
            if self.arrival_index == self.arrivals_sampled:
                self.arrival_times = sample_arrivals(self.lam_r, self.arrivals_sampled)
                self.arrival_index = 0
            self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + t].append(agent_id)
            self.arrival_index += 1

        new_matched = market.step()
        for matched_order in new_matched:
            agent_id = matched_order.order.agent_id
            quantity = matched_order.order.order_type*matched_order.order.quantity
            cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
            self.agents[agent_id].update_position(quantity, cash)

        for ctrl in self.controllers:
            ctrl.observe(t, new_matched)
        if self.lob.enabled:
            self.lob.record(market, t, pre, new_matched, posts=posts, lookup=self.wash_meta)
