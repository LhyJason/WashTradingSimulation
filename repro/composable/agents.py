r"""
agents.py -- agent registry + calling-convention adapter + builders

Goal: normalize the various upstream agents into a form the matching loop can call
       directly, so we can freely compose agent types and counts **without modifying
       any upstream code**.

Why an adapter is needed (see paper §3.1 Action Phase and this repo's code):
  The upstream SimulatorSampledArrival.step() calls each agent with a hard-coded,
  no-argument `agent.take_action()`, and the agent itself flips a coin for buy/sell
  (paper §4: "Whether they buy or sell is determined by a fair coin flip.").
  But different agents do not share the same take_action signature:
    - ZIAgent.take_action()          no arg, picks side via random.choice   -- convention (1)
    - HBLAgent.take_action(side, ..) needs BUY/SELL passed in from outside   -- convention (2)
  AgentActionAdapter wraps convention-(2) agents: it exposes a no-arg take_action()
  that flips a fair coin for the side internally and then forwards. This way the
  upstream step() needs no change at all.
"""
import random

from marketsim.fourheap.constants import BUY, SELL
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.hbl_agent import HBLAgent
# NOTE: informed_ZI.py and noise_ZI_agent.py BOTH define a class literally named `ZIAgent`,
# colliding with the plain ZIAgent above -- so we import them under aliases.
from marketsim.agent.informed_ZI import ZIAgent as InformedZIAgent
from marketsim.agent.noise_ZI_agent import ZIAgent as NoiseZIAgent


# --------------------------------------------------------------------------- #
# Adapter: normalize every agent to the "no-arg take_action()" convention
# --------------------------------------------------------------------------- #
class AgentActionAdapter:
    """Lightweight wrapper: exposes a no-arg take_action(); all other attributes/methods
    are forwarded to the real agent.

    For needs_side=True agents (e.g. HBL), on each arrival it flips a fair coin for
    BUY/SELL and calls the real agent's take_action(side); for needs_side=False (e.g. ZI)
    it forwards directly and the agent flips its own coin. In both cases the side is a
    fair random draw, matching paper §4.
    """

    def __init__(self, agent, needs_side: bool):
        # Write straight into __dict__ so __getattr__ never recurses on these two names
        self._agent = agent
        self._needs_side = needs_side

    def take_action(self):
        if self._needs_side:
            side = random.choice([BUY, SELL])
            return self._agent.take_action(side)
        return self._agent.take_action()

    # Everything else (position / cash / get_pos_value / update_position / reset /
    # get_id / agent_id ...) is forwarded to the wrapped real agent
    def __getattr__(self, name):
        return getattr(self._agent, name)

    def __str__(self):
        return f"Adapter({self._agent})"


# --------------------------------------------------------------------------- #
# Agent registry: how to construct each type "correctly", and which calling
# convention it follows
# --------------------------------------------------------------------------- #
# `common` is the dict of environment/market parameters shared by all agents
# (injected by the simulator); `params` are the per-spec overrides for this type.
# Each builder returns a real agent instance.

def _build_zi(agent_id, market, common, params):
    # ZIAgent(agent_id, market, q_max, shade, pv_var, eta=1.0)  -- paper §4.1
    return ZIAgent(
        agent_id=agent_id,
        market=market,
        q_max=params.get("q_max", common["q_max"]),
        shade=params.get("shade", common["shade"]),
        pv_var=params.get("pv_var", common["pv_var"]),
        eta=params.get("eta", common["eta"]),
    )


def _build_hbl(agent_id, market, common, params):
    # HBLAgent(agent_id, market, q_max, shade, L, pv_var, arrival_rate, pv) -- paper §4.2
    # Note: pv=-1 MUST be passed explicitly, otherwise the upstream constructor sets
    # self.pv to None and crashes. arrival_rate drives the grace period tau_gp = 1/lambda_a
    # (paper §4.2); it defaults to the environment's lambda.
    return HBLAgent(
        agent_id=agent_id,
        market=market,
        q_max=params.get("q_max", common["q_max"]),
        shade=params.get("shade", common["shade"]),
        pv_var=params.get("pv_var", common["pv_var"]),
        L=params.get("L", common["L"]),
        arrival_rate=params.get("arrival_rate", common["lam"]),
        pv=-1,
    )


def _build_informed_zi(agent_id, market, common, params):
    # informed_ZI.ZIAgent(agent_id, market, q_max, shade, pv_var)  -- imported as InformedZIAgent
    # "Informed": prices off the FINAL fundamental directly (effectively knows f_T) instead
    # of estimating it from the current observation. take_action(side) -> convention (2).
    return InformedZIAgent(
        agent_id=agent_id,
        market=market,
        q_max=params.get("q_max", common["q_max"]),
        shade=params.get("shade", common["shade"]),
        pv_var=params.get("pv_var", common["pv_var"]),
    )


def _build_noise_zi(agent_id, market, common, params):
    # noise_ZI_agent.ZIAgent(agent_id, market, q_max, shade, pv_var, est_var) -- imported as NoiseZIAgent
    # "Noisy": adds Gaussian noise N(0, est_var) to its fundamental estimate.
    # est_var has no environment-wide default, so it falls back to 1e6 here; override it
    # per-spec via AgentSpec("NoiseZI", n, {"est_var": ...}). take_action(side) -> convention (2).
    return NoiseZIAgent(
        agent_id=agent_id,
        market=market,
        q_max=params.get("q_max", common["q_max"]),
        shade=params.get("shade", common["shade"]),
        pv_var=params.get("pv_var", common["pv_var"]),
        est_var=params.get("est_var", common.get("est_var", 1e6)),
    )


# type name -> (builder, needs_side)
AGENT_REGISTRY = {
    "ZI":         (_build_zi,          False),  # convention (1): no-arg take_action, flips its own coin
    "HBL":        (_build_hbl,         True),   # convention (2): needs a side passed in
    "InformedZI": (_build_informed_zi, True),   # convention (2): ZI that knows the final fundamental
    "NoiseZI":    (_build_noise_zi,    True),   # convention (2): ZI with a noisy fundamental estimate
}


def make_agent(agent_type: str, agent_id: int, market, common: dict, params: dict):
    """Construct a real agent by type name and return it wrapped in the adapter."""
    if agent_type not in AGENT_REGISTRY:
        raise KeyError(
            f"Unknown agent type {agent_type!r}; registered: {list(AGENT_REGISTRY)}"
        )
    builder, needs_side = AGENT_REGISTRY[agent_type]
    real_agent = builder(agent_id, market, common, params)
    return AgentActionAdapter(real_agent, needs_side)
