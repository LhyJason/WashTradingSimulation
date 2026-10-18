r"""export.py -- v7 run export: v6's layout and modes, plus the round labels.

Same folders, files and columns as ``washtrade/v6/export.py`` (detection reads v7 runs
unchanged). v7 changes:

* ``SCHEMA_VERSION = "washtrade.v7/1"``; meta.json ``summary`` / ``groups`` are v7's metrics
  (analysis.py: rotation, pair repeats, residuals, zero returns).
* ``labels/wash_events.csv`` gains ``round_idx``, ``unit_idx``, ``piece`` (and ``edge_idx`` is
  the edge's position inside its round).
* new ``labels/wash_rounds.csv``: one row per round -- members (wallet codes, in cycle order),
  size, residual, number of events, planned volume, open / close time.
* ``labels/wallets.csv`` gains ``rounds_used`` (rounds the wallet was drawn into; empty for
  non-wash wallets). Pool wallets that were never drawn have ``rounds_used = 0`` and no legs.
"""
import json
import os
import shutil
import time

import pandas as pd

from washtrade.v3.export import load_labels, load_meta, wallet_codes  # noqa: F401
from washtrade.v6.export import EXPORT_MODES, is_complete, load_observable  # noqa: F401
from washtrade.v6.export import _listing
from washtrade.v6.export import export_run as _v6_export_run

from . import analysis as A

SCHEMA_VERSION = "washtrade.v7/1"


def export_run(sim, run_dir, *, run_id, seed, scenario=None, background=None,
               background_spec=None, factors=None, mode="full", anonymize=True,
               extra_meta=None) -> dict:
    meta = dict(schema_version=SCHEMA_VERSION, summary=A.summarize(sim),
                groups=A.group_summary(sim).to_dict(orient="records"))
    if extra_meta:
        meta.update(extra_meta)
    paths = _v6_export_run(sim, run_dir, run_id=run_id, seed=seed, scenario=scenario,
                           background=background, background_spec=background_spec,
                           factors=factors, mode=mode, anonymize=anonymize, extra_meta=meta)

    codes = wallet_codes(sim, seed, anonymize)
    lab_dir = os.path.join(run_dir, "labels")

    wp = paths["labels/wallets.csv"]
    w = pd.read_csv(wp)
    used = {aid: n for c in sim.controllers for aid, n in c.rounds_used.items()}
    w["rounds_used"] = w["agent_id"].map(used).astype("Int64")
    w.to_csv(wp, index=False)

    rounds = A.round_log_df(sim)
    if len(rounds):
        rounds["members"] = [";".join(codes[int(a)] for a in m.split(";"))
                             for m in rounds["members"]]
    rp = os.path.join(lab_dir, "wash_rounds.csv")
    rounds.to_csv(rp, index=False)
    paths["labels/wash_rounds.csv"] = rp

    mp = paths["meta.json"]
    with open(mp, encoding="utf-8") as fh:
        m = json.load(fh)
    m["files"] = sorted(k for k in paths if k != "meta.json")
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2, ensure_ascii=False, default=str)
    return paths


def export_run_atomic(sim, run_dir, **kw) -> dict:
    """``export_run`` into ``<run_dir>.tmp``, then rename to ``run_dir`` (retried on Windows,
    where a scanner or indexer can briefly lock just-written files)."""
    tmp = run_dir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    export_run(sim, tmp, **kw)
    for attempt in range(60):
        try:
            os.replace(tmp, run_dir)
            break
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.05 * (attempt + 1))
    return {rel: os.path.join(run_dir, rel) for rel in _listing(run_dir)}
