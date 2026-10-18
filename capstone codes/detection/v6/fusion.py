r"""fusion.py -- v6's channel and method lists on v4's machinery, plus where the cluster channel applies.

The cluster channel joins v4's seven channels under v4's rules, unchanged: control-run tail
probability u_k, floor 0.5 (pre-declared in v4: a score only counts once it reaches 0.5), max-
evidence fusion, exact thresholds at a pooled control FPR of 1% (0.1% and 5% as sensitivity),
floors 0, 0.3 and 0.7 as sensitivity.

Methods: every v4 method, the cluster channel alone at each resolution, and

  * ``v6``     = v4's seven channels + ``cluster`` (resolution 1): the main detector;
  * ``v6_r2``  = v4's seven channels + ``cluster_r2``  (resolution 2, sensitivity).

**Applicability.** The cluster channel is used in a background only if its control runs mix: mean
modularity of the resolution-1 partitions of the control runs below MAX_CONTROL_MODULARITY (see
cluster.py, 4.). Elsewhere every cluster score is replaced by 0 before scoring, so it never reaches
the floor: the cluster methods flag nothing and ``v6`` equals ``v4`` exactly. The rule uses only
control runs, no labels.
"""
import numpy as np
import pandas as pd

from detection.v4.detector import CHANNELS
from detection.v4.fusion import FUSIONS as V4_FUSIONS
from detection.v4.fusion import MAIN_FLOOR
from detection.v4.fusion import SINGLE as V4_SINGLE
from detection.v4.fusion import Null as _Null

from .cluster import MAX_CONTROL_MODULARITY, RESOLUTIONS, channel_name

CLUSTER = [channel_name(r) for r in RESOLUTIONS]
MODULARITY = f"modularity_{channel_name(1.0)}"      # run-level column carried by the control pairs
SINGLE = V4_SINGLE + CLUSTER
FUSIONS = dict(V4_FUSIONS)
FUSIONS["v6"] = list(CHANNELS) + [channel_name(1.0)]
FUSIONS["v6_r2"] = list(CHANNELS) + [channel_name(2.0)]
METHODS = SINGLE + list(FUSIONS)


def applicability(control: pd.DataFrame) -> pd.DataFrame:
    """Per background: mean control modularity (one value per control run) and whether the
    cluster channel is used."""
    runs = control.drop_duplicates("run_id")
    out = runs.groupby("background")[MODULARITY].agg(["mean", "size"]).rename(
        columns={"mean": "control_modularity", "size": "n_control_runs"})
    out["cluster_enabled"] = out["control_modularity"] < MAX_CONTROL_MODULARITY
    return out.reset_index()


class Null(_Null):
    """v4's Null over v6's channels and fusions, with the cluster channel switched off where the
    control runs do not mix."""

    def __init__(self, control: pd.DataFrame):
        super().__init__(control, singles=SINGLE, fusions=FUSIONS)
        app = applicability(control)
        self.enabled = dict(zip(app["background"], app["cluster_enabled"]))

    def gate(self, bg, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.enabled.get(bg, False):
            return pairs
        off = {c: np.where(np.isnan(pairs[c].to_numpy(float)), np.nan, 0.0) for c in CLUSTER}
        return pairs.assign(**off)

    def fused(self, bg, pairs, channels, floor=MAIN_FLOOR):
        return super().fused(bg, self.gate(bg, pairs), channels, floor)

    def method_scores(self, bg, pairs, floor=MAIN_FLOOR):
        return super().method_scores(bg, self.gate(bg, pairs), floor)
