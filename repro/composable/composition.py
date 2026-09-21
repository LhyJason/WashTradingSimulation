r"""
composition.py -- describes "which agents are in a simulation, how many of each,
and their parameters"

Two equivalent ways to write it; both get normalized into the same List[AgentSpec]:

  1) Simple-label form (easiest, parameters fall back to environment defaults):
        {"ZI": 13, "HBL": 2}

  2) AgentSpec form (fine-grained control, can give a type custom parameters):
        [AgentSpec("ZI", 13, {"eta": 0.2}),
         AgentSpec("HBL", 2, {"L": 4})]

normalize_composition() unifies either of the above into a List[AgentSpec].
"""
from dataclasses import dataclass, field


@dataclass
class AgentSpec:
    """A group of same-type agents: type name + count + per-type parameter overrides (optional)."""
    agent_type: str
    count: int
    params: dict = field(default_factory=dict)


def normalize_composition(composition):
    """Unify a dict- or list-form composition into a List[AgentSpec].

    - dict:  {"ZI": 13, "HBL": 2}            -> use default parameters
    - list:  [AgentSpec(...), ("ZI", 13), ("HBL", 2, {...})]
             list items may be AgentSpec, or a (type, count) / (type, count, params)
             tuple for quick shorthand.
    """
    if isinstance(composition, dict):
        return [AgentSpec(t, int(n)) for t, n in composition.items()]

    specs = []
    for item in composition:
        if isinstance(item, AgentSpec):
            specs.append(item)
        elif isinstance(item, (tuple, list)):
            if len(item) == 2:
                t, n = item
                specs.append(AgentSpec(t, int(n)))
            elif len(item) == 3:
                t, n, p = item
                specs.append(AgentSpec(t, int(n), dict(p)))
            else:
                raise ValueError(f"Cannot parse composition item: {item!r}")
        else:
            raise TypeError(f"Unsupported composition item type: {type(item)} -> {item!r}")
    return specs


def total_agents(specs):
    """Total number of agents in the composition."""
    return sum(s.count for s in specs)
