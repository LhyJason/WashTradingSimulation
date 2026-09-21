"""composable -- a freely composable (agent types and counts) simulation environment built on PyMarketSim, without modifying upstream code."""
from . import _patches  # noqa: F401  runtime fix for the upstream Order.__eq__ None-comparison bug (originals untouched)
from .composition import AgentSpec, normalize_composition, total_agents
from .agents import AgentActionAdapter, make_agent, AGENT_REGISTRY
from .simulator import ComposableSimulator

__all__ = [
    "AgentSpec",
    "normalize_composition",
    "total_agents",
    "AgentActionAdapter",
    "make_agent",
    "AGENT_REGISTRY",
    "ComposableSimulator",
]
