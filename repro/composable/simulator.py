r"""
simulator.py -- ComposableSimulator: a simulator with freely composable agents

Design points:
  * Inherits the upstream marketsim.simulator.sampled_arrival_simulator.SimulatorSampledArrival
    and **fully reuses its step() / run() / end_sim()** (the real matching + arrival
    scheduling + settlement logic).
  * Overrides only __init__: replaces the upstream hard-coded "build N+1 identical ZI"
    with "build arbitrary agents from a composition spec".
  * The upstream step() calls each agent as a no-arg take_action(); we wrap every agent
    with AgentActionAdapter, so step() needs no change at all -- no upstream code modified.

Correspondence with the paper (§3.1 / §3.4 / §5.2):
  * Fundamental value f_t: mean-reverting process LazyGaussianMeanReverting (paper Eq. (1)(2)).
  * Arrivals: each agent samples its next arrival time from a geometric distribution
    (paper §3.1 "geometric distribution").
  * Fixed environment parameters default to Table 3: q_max=10, f_bar=1e5, kappa=0.01, T=2000.
"""
from collections import defaultdict

from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.simulator.sampled_arrival_simulator import (
    SimulatorSampledArrival,
    sample_arrivals,
)

from .agents import make_agent
from .composition import normalize_composition, total_agents


class ComposableSimulator(SimulatorSampledArrival):
    def __init__(self,
                 agent_composition,            # dict {"ZI":13,"HBL":2} or List[AgentSpec]
                 sim_time: int = 2000,         # paper T
                 num_assets: int = 1,          # paper K (single market here)
                 lam: float = 5e-3,            # paper lambda (reentry rate)
                 mean: float = 1e5,            # paper f_bar
                 r: float = 0.01,              # paper kappa (mean reversion; called r in code)
                 shock_var: float = 1e6,       # paper sigma_s^2
                 q_max: int = 10,              # paper q_max
                 pv_var: float = 5e6,          # paper sigma_pv^2
                 shade=None,                   # [R_min, R_max] for ZI/HBL
                 eta: float = 0.2,             # ZI threshold eta (paper §4.1)
                 L: int = 4,                   # HBL memory length (paper §4.2)
                 lam_r: float = None):         # reentry arrival rate (defaults to lam)

        if shade is None:
            shade = [250, 500]
        if lam_r is None:
            lam_r = lam

        # ---- Parse the composition ----
        specs = normalize_composition(agent_composition)
        n_total = total_agents(specs)

        # ---- Set every state attribute the upstream step()/run()/end_sim() relies on ----
        self.num_agents = n_total
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.lam_r = lam_r
        self.time = 0
        self.hbl_agent = False  # switch used by the upstream __init__; we don't take its branch

        # Arrival scheduling: as upstream, pre-sample a large batch of geometric inter-arrivals
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(lam_r, self.arrivals_sampled)
        self.arrival_index = 0

        # ---- Build the markets (identical to upstream: mean-reverting fundamental + Market) ----
        self.markets = []
        for _ in range(num_assets):
            fundamental = LazyGaussianMeanReverting(
                mean=mean, final_time=sim_time, r=r, shock_var=shock_var
            )
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        # ---- Shared parameters, injected into each agent builder ----
        common = dict(q_max=q_max, pv_var=pv_var, shade=shade,
                      eta=eta, L=L, lam=lam)

        # ---- Build agents in spec order, assign consecutive ids, sample a first arrival each ----
        self.agents = {}
        self.agent_types = {}   # agent_id -> type name (for per-type profit stats)
        agent_id = 0
        for spec in specs:
            for _ in range(spec.count):
                # First arrival time: as upstream, take the next from arrival_times
                self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
                self.arrival_index += 1

                self.agents[agent_id] = make_agent(
                    agent_type=spec.agent_type,
                    agent_id=agent_id,
                    market=self.markets[0],
                    common=common,
                    params=spec.params,
                )
                self.agent_types[agent_id] = spec.agent_type
                agent_id += 1

    # step() / run() / end_sim() are all inherited from upstream; not overridden.

    def final_values(self):
        """After a run, return {agent_id: final wealth} per paper Eq. (4): cash + q_T*f_T + private values."""
        f_T = self.markets[0].get_final_fundamental()
        values = {}
        for aid, agent in self.agents.items():
            values[aid] = float(agent.get_pos_value()) + agent.position * f_T + agent.cash
        return values
