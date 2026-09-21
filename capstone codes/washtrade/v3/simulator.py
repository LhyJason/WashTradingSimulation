r"""simulator.py -- WashTradingSimulator (v3): background market + any number of wash groups.

Relationship to upstream: v1/v2 delegated each timestep to the inherited ``step()``. v3 has to
re-state that step body, for ONE reason: upstream cancels an arriving agent's orders with
``market.withdraw_all(agent_id)``. A camouflage wallet is both a ZI (it arrives) and a wash
wallet (it may have an exposed wash leg resting), so withdraw-all would let its own ZI arrival
cancel its wash leg. v3 replaces that single call with order-scoped cancellation:

    ZI arrival       -> cancel this agent's NON-wash orders   (order_id <  1e9)
    wash leg posted  -> cancel this wallet's stale WASH orders (order_id >= 1e9)

For every agent that is not a camouflage wallet the two coincide with withdraw-all, and the rest
of the step body (arrival resampling, batch clear, position/cash booking) is copied verbatim
from ``SimulatorSampledArrival.step``. Nothing in marketsim/ or repro/composable/ is modified.

Wash groups:

    WashTradingSimulator({"ZI": 13}, wash_groups=[
        WashGroup(n_wallets=3, gap=0),
        WashGroup(n_wallets=2, gap=5, camouflage=True),
    ], wash_seed=0, sim_time=2000, lam=5e-3)

The v2 calling convention (``{"ZI": 13, "WashTrader": 3}`` + ``wash_gap=`` etc.) still works and
builds a single group -- that is what the v2-equivalence check uses.
"""
import inspect
from dataclasses import asdict, dataclass
from typing import Optional

from marketsim.simulator.sampled_arrival_simulator import sample_arrivals
from repro.composable.agents import make_agent
from repro.composable.composition import normalize_composition
from repro.composable.simulator import ComposableSimulator

from .controller import WashController, is_wash_oid
from .lob import LOBRecorder
from .wash_agent import WashTrader


@dataclass
class WashGroup:
    """One colluding wash group (one controller)."""
    n_wallets: int = 3
    gap: int = 0                      # timesteps between leg 1 and leg 2
    lam_wash: Optional[float] = None  # event rate; None -> the market's lam
    q: int = 1
    eps: float = 0.0                  # keep 0 (v1 finding)
    first_leg: str = "random"
    camouflage: bool = False          # wallets also trade as ordinary background agents
    camouflage_type: str = "ZI"       # which background behaviour they camouflage as
    seed: Optional[int] = None        # None -> wash_seed + 1000 * group index


def _composable_defaults():
    """ComposableSimulator's own keyword defaults (q_max, pv_var, shade, eta, L, ...), read from
    its signature so camouflage agents are built with exactly the background's parameters."""
    sig = inspect.signature(ComposableSimulator.__init__)
    return {k: p.default for k, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty}


class WashTradingSimulator(ComposableSimulator):
    WASH_TYPE = "WashTrader"

    def __init__(self, agent_composition, *, wash_groups=None,
                 wash_q=1, wash_eps=0.0, wash_gap=0, wash_first_leg="random",
                 lam_wash=None, wash_seed=None, n_wallets=None, camouflage=False,
                 record_lob=True, lob_depth_levels=5, **kwargs):
        specs = normalize_composition(agent_composition)
        wash_count = sum(s.count for s in specs if s.agent_type == self.WASH_TYPE)
        background = [s for s in specs if s.agent_type != self.WASH_TYPE]

        if wash_groups is None:                       # v2 calling convention -> <= 1 group
            n = wash_count if n_wallets is None else n_wallets
            wash_groups = ([WashGroup(n_wallets=n, gap=wash_gap, lam_wash=lam_wash, q=wash_q,
                                      eps=wash_eps, first_leg=wash_first_leg,
                                      camouflage=camouflage)] if n > 0 else [])
        elif wash_count:
            raise ValueError("give wash wallets either as 'WashTrader' in the composition "
                             "or via wash_groups, not both")
        groups = [g if isinstance(g, WashGroup) else WashGroup(**g) for g in wash_groups]

        super().__init__(background, **kwargs)
        market = self.markets[0]

        defaults = _composable_defaults()
        common = {k: kwargs.get(k, defaults[k]) for k in ("q_max", "pv_var", "shade", "eta", "L")}
        if common["shade"] is None:
            common["shade"] = [250, 500]              # same fallback as ComposableSimulator
        common["lam"] = self.lam

        self.groups = groups
        self.controllers = []
        self.wash_wallet_ids = []
        self.camouflage_ids = []
        self.wallet_group = {}                        # agent_id -> group index
        next_id = len(self.agents)
        for k, g in enumerate(groups):
            if g.n_wallets < 2:
                raise ValueError(f"wash group {k} needs >= 2 wallets, got {g.n_wallets}")
            ids = []
            for _ in range(g.n_wallets):
                aid = next_id
                next_id += 1
                if g.camouflage:
                    # an ordinary background trader: own private values + geometric arrivals,
                    # first arrival drawn exactly like ComposableSimulator draws them
                    self.agents[aid] = make_agent(g.camouflage_type, aid, market, common, {})
                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = sample_arrivals(self.lam_r, self.arrivals_sampled)
                        self.arrival_index = 0
                    self.arrivals[self.arrival_times[self.arrival_index].item()].append(aid)
                    self.arrival_index += 1
                    self.camouflage_ids.append(aid)
                else:
                    self.agents[aid] = WashTrader(aid, market)
                self.agent_types[aid] = self.WASH_TYPE
                self.wallet_group[aid] = k
                ids.append(aid)
            self.wash_wallet_ids.extend(ids)

            seed = g.seed if g.seed is not None else (
                None if wash_seed is None else int(wash_seed) + 1000 * k)
            self.controllers.append(WashController(
                wallets=[self.agents[i] for i in ids], market=market,
                q=g.q, eps=g.eps, lam_wash=(g.lam_wash if g.lam_wash is not None else self.lam),
                gap=g.gap, first_leg=g.first_leg, seed=seed, sim_time=self.sim_time,
                group_id=k,
            ))
        self.num_agents = len(self.agents)

        self.lob = LOBRecorder(depth_levels=lob_depth_levels, enabled=record_lob)

        self.run_config = dict(kwargs)
        self.run_config.update(
            composition=[(s.agent_type, s.count) for s in background],
            wash_groups=[asdict(g) for g in groups], wash_seed=wash_seed,
            record_lob=record_lob, lob_depth_levels=lob_depth_levels,
            sim_time=self.sim_time, lam=self.lam, num_agents=self.num_agents,
            wash_wallet_ids=list(self.wash_wallet_ids),
            camouflage_ids=list(self.camouflage_ids),
        )

    # -- convenience --------------------------------------------------------- #
    @property
    def controller(self):
        """The single controller (v2-style access). Ambiguous with several groups."""
        if len(self.controllers) != 1:
            raise AttributeError(f"{len(self.controllers)} wash groups; use sim.controllers")
        return self.controllers[0]

    def wash_meta(self, order_id):
        """Metadata of a wash order_id (from whichever group owns it), else None."""
        if not is_wash_oid(order_id):
            return None
        k = (int(order_id) - WashController._OID_BASE) // WashController._OID_GROUP_STRIDE
        if 0 <= k < len(self.controllers):
            return self.controllers[k].oid_meta.get(order_id)
        return None

    # -- order-scoped cancellation ------------------------------------------ #
    def _cancel(self, agent_id, wash: bool):
        """Cancel ``agent_id``'s resting WASH orders (wash=True) or NON-wash orders (wash=False).

        Mirrors ``FourHeap.withdraw_all`` (same guard, same remove() calls, same list reset) --
        for an agent whose orders are all of the cancelled kind it IS withdraw_all."""
        ob = self.markets[0].order_book
        if agent_id in ob.agent_id_map and ob.agent_id_map[agent_id]:
            kept = []
            for order_id in ob.agent_id_map[agent_id]:
                if is_wash_oid(order_id) == wash:
                    ob.remove(order_id)
                elif ob.buy_unmatched.contains(order_id) or ob.sell_unmatched.contains(order_id):
                    kept.append(order_id)             # still resting -> keep tracking it
            ob.agent_id_map[agent_id] = kept

    # -- scheduling ---------------------------------------------------------- #
    def _posting_controllers(self, t):
        """Groups that post at t. Every due leg-2 (close) goes; an event START goes only if no
        other group posts in this timestep, otherwise it is deferred by one step."""
        due = [c for c in self.controllers if c.due(t)]
        posting = [c for c in due if c.pending()]
        for c in due:
            if c.pending():
                continue
            if posting:
                c.defer_open()
            else:
                posting.append(c)
        return posting

    # -- simulation loop ----------------------------------------------------- #
    def step(self):
        t = self.time
        if t >= self.sim_time:
            return self.end_sim()
        market = self.markets[0]
        pre = self.lob.book_state(market) if self.lob.enabled else None
        market.event_queue.set_time(t)

        # 1) wash legs
        posts = []
        for ctrl in self._posting_controllers(t):
            for aid, order in ctrl.orders_for(t):
                self._cancel(aid, wash=True)
                market.add_orders([order])
                posts.append((ctrl.group_id, aid, order))

        # 2) background (and camouflage) arrivals -- SimulatorSampledArrival.step, except that
        #    withdraw_all(agent_id) is replaced by the order-scoped cancel above
        for agent_id in self.arrivals[t]:
            agent = self.agents[agent_id]
            self._cancel(agent_id, wash=False)
            orders = agent.take_action()
            market.add_orders(orders)
            if self.arrival_index == self.arrivals_sampled:
                self.arrival_times = sample_arrivals(self.lam_r, self.arrivals_sampled)
                self.arrival_index = 0
            self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + t].append(agent_id)
            self.arrival_index += 1

        # 3) the single batch clear + position/cash booking (verbatim upstream formulas)
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

    def run(self):
        for t in range(self.sim_time):
            if self.arrivals[t] or any(c.due(t) for c in self.controllers):
                self.step()
            self.time += 1
        self.step()
        for ctrl in self.controllers:
            ctrl.finalize()
