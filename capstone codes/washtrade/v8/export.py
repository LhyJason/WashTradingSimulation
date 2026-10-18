r"""export.py -- v8 run export: v7's layout, files and columns, with v8's metrics.

Same folders, files and columns as ``washtrade/v7/export.py``, so detection reads v8 runs
unchanged. Only ``meta.json`` differs: ``schema_version = "washtrade.v8/1"``, and ``summary`` /
``groups`` hold v8's metrics (analysis.py: execution, repeat share, hold times, wash volume,
honest order sizes). ``labels/wash_rounds.csv`` lists rounds as v7 does; with a pipeline, rounds
overlap in time.
"""
from washtrade.v7.export import (  # noqa: F401  (re-exported)
    EXPORT_MODES, is_complete, load_labels, load_meta, load_observable, wallet_codes,
)
from washtrade.v7.export import export_run as _v7_export_run
from washtrade.v7.export import export_run_atomic as _v7_export_run_atomic

from . import analysis as A

SCHEMA_VERSION = "washtrade.v8/1"


def _v8_meta(sim, extra_meta):
    meta = dict(schema_version=SCHEMA_VERSION, summary=A.summarize(sim),
                groups=A.group_summary(sim).to_dict(orient="records"))
    meta.update(extra_meta or {})
    return meta


def export_run(sim, run_dir, *, extra_meta=None, **kw) -> dict:
    """v7's export; v7 lets ``extra_meta`` override its summary, groups and schema version."""
    return _v7_export_run(sim, run_dir, extra_meta=_v8_meta(sim, extra_meta), **kw)


def export_run_atomic(sim, run_dir, *, extra_meta=None, **kw) -> dict:
    return _v7_export_run_atomic(sim, run_dir, extra_meta=_v8_meta(sim, extra_meta), **kw)
