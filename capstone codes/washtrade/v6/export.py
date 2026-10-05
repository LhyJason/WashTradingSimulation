r"""export.py -- v6 run export: v3's layout, three size modes, crash-safe writes.

The folder layout and every CSV column of ``washtrade/v3/export.py`` are kept, so detection/v1
reads v6 runs unchanged. v6 changes:

* ``SCHEMA_VERSION = "washtrade.v6/1"``; meta.json also carries the background spec, the design
  factors of the run (``factors``) and v6's summary / per-group metrics (leakage, camouflage
  share, background activity).
* ``labels/wallets.csv`` gains ``camouflage_lam`` (own arrival rate; empty = background rate).
* ``mode``:  "full"  observable/{trades, lob, lob_levels} + labels  (= v3)
             "slim"  no lob_levels.csv                              (~1/3 of the size)
             "tape"  no LOB files at all (trades + labels); simulate with record_lob=False
* ``export_run_atomic`` writes into ``<run_dir>.tmp`` and renames it when complete, so a job
  killed mid-write never leaves a run folder that looks finished. ``meta.json`` is the
  completion marker that ``generate.py`` uses to skip finished runs.
"""
import json
import os
import shutil
import time

import pandas as pd

from washtrade.v3.export import export_run as _v3_export_run
from washtrade.v3.export import load_labels, load_meta  # noqa: F401

from . import analysis as A

SCHEMA_VERSION = "washtrade.v6/1"
EXPORT_MODES = ("full", "slim", "tape")


def export_run(sim, run_dir, *, run_id, seed, scenario=None, background=None,
               background_spec=None, factors=None, mode="full", anonymize=True,
               extra_meta=None) -> dict:
    if mode not in EXPORT_MODES:
        raise ValueError(f"mode must be one of {EXPORT_MODES}, got {mode!r}")
    if mode == "tape" and sim.lob.enabled:
        raise ValueError("mode='tape' expects a run simulated with record_lob=False")
    if mode != "tape" and not sim.lob.enabled:
        raise ValueError(f"mode={mode!r} needs the LOB; simulate with record_lob=True")

    meta_extra = dict(
        schema_version=SCHEMA_VERSION, export_mode=mode,
        background_spec=background_spec, factors=factors or {},
        summary=A.summarize(sim),
        groups=A.group_summary(sim).to_dict(orient="records"),
        background_legs_by_bin=A.background_activity(sim)["background_legs_by_bin"],
    )
    if extra_meta:
        meta_extra.update(extra_meta)
    paths = _v3_export_run(sim, run_dir, run_id=run_id, seed=seed, scenario=scenario,
                           background=background, anonymize=anonymize,
                           save_levels=(mode == "full"), extra_meta=meta_extra)

    # labels/wallets.csv: add each wallet's own camouflage arrival rate
    wp = paths["labels/wallets.csv"]
    w = pd.read_csv(wp)
    rate = {aid: own.lam for aid, own in sim.own_arrivals.items()}
    w["camouflage_lam"] = w["agent_id"].map(rate)
    w.to_csv(wp, index=False)

    if mode == "tape":
        for rel in ("observable/lob.csv", "labels/lob_latent.csv"):
            os.remove(paths.pop(rel))
        mp = paths["meta.json"]
        with open(mp, encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["files"] = sorted(k for k in paths if k != "meta.json")
        with open(mp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    return paths


def export_run_atomic(sim, run_dir, **kw) -> dict:
    """``export_run`` into ``<run_dir>.tmp``, then rename to ``run_dir``."""
    tmp = run_dir + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    export_run(sim, tmp, **kw)
    # On Windows a just-written file can be held open briefly (virus scanner, search indexer),
    # which makes the folder rename fail with PermissionError; retry with a growing pause.
    for attempt in range(60):
        try:
            os.replace(tmp, run_dir)
            break
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.05 * (attempt + 1))
    return {rel: os.path.join(run_dir, rel) for rel in _listing(run_dir)}


def _listing(run_dir):
    out = []
    for folder, _, files in os.walk(run_dir):
        for f in files:
            out.append(os.path.relpath(os.path.join(folder, f), run_dir).replace(os.sep, "/"))
    return sorted(out)


def load_observable(run_dir) -> dict:
    """What a detector gets: {'trades', 'lob' and 'lob_levels' if exported}."""
    obs = os.path.join(run_dir, "observable")
    out = {"trades": pd.read_csv(os.path.join(obs, "trades.csv"))}
    for name in ("lob", "lob_levels"):
        p = os.path.join(obs, name + ".csv")
        if os.path.exists(p):
            out[name] = pd.read_csv(p)
    return out


def is_complete(run_dir) -> bool:
    return os.path.exists(os.path.join(run_dir, "meta.json"))
