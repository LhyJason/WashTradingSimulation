r"""generate.py -- build a labelled wash-trading dataset (layout of export.py) in one command.

Run (from the repo root):
  python "capstone codes/washtrade/v4/generate.py" --name ds_v4 --seeds 10

Layout written:
  <out>/<name>/
    dataset.json                 parameters, schema version, creation time
    manifest.csv                 one row per run: ids, scenario, background, seed, headline stats
    runs/<background>__<scenario>__s<seed>/   observable/ labels/ meta.json  (see export.py)

Defaults: all scenarios x {thin, thick} x seeds 0..9, T = 2000 -> 180 runs.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))     # .../capstone codes
_ROOT = os.path.dirname(_CAP)                       # .../<repo root>
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from washtrade.v3 import analysis as A
from washtrade.v4 import scenarios as S
from washtrade.v4.export import SCHEMA_VERSION, export_run

DEFAULT_OUT = os.path.join(_HERE, "data")


def generate(name, *, seeds=10, backgrounds=("thin", "thick"), scenarios=None,
             out_dir=None, sim_time=2000, save_levels=True, anonymize=True, verbose=True):
    """Simulate + export every (background, scenario, seed). Returns the manifest DataFrame."""
    seed_list = list(range(seeds)) if isinstance(seeds, int) else [int(s) for s in seeds]
    scenarios = list(S.SCENARIOS) if scenarios is None else list(scenarios)
    root = os.path.join(out_dir or DEFAULT_OUT, name)
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)

    rows = []
    total = len(backgrounds) * len(scenarios) * len(seed_list)
    t0 = time.time()
    for bg in backgrounds:
        for sc in scenarios:
            for seed in seed_list:
                rid = S.run_id(bg, sc, seed)
                rdir = os.path.join(root, "runs", rid)
                sim = S.run(bg, sc, seed, sim_time=sim_time, record_lob=True)
                export_run(sim, rdir, run_id=rid, seed=seed, scenario=sc, background=bg,
                           anonymize=anonymize, save_levels=save_levels)
                s = A.summarize(sim)
                rows.append(dict(
                    run_id=rid, background=bg, scenario=sc, seed=seed,
                    path=os.path.relpath(rdir, root).replace(os.sep, "/"),
                    n_groups=s["n_groups"], n_wallets=s["n_wallets"],
                    n_camouflage_wallets=s["n_camouflage_wallets"], n_background=s["n_background"],
                    n_events=s["n_events"], self_trade_rate=s["self_trade_rate"],
                    intercept_rate=s["intercept_rate"], total_trade_legs=s["total_trade_legs"],
                    total_volume=s["total_volume"],
                    wash_leg_volume_share=s["wash_leg_volume_share"],
                ))
                if verbose and (len(rows) % 20 == 0 or len(rows) == total):
                    print(f"  {len(rows):4d}/{total} runs  ({time.time() - t0:5.1f}s)")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(root, "manifest.csv"), index=False)
    with open(os.path.join(root, "dataset.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(name=name, schema_version=SCHEMA_VERSION,
                       created=datetime.now().isoformat(timespec="seconds"),
                       seeds=seed_list, backgrounds=list(backgrounds), scenarios=scenarios,
                       sim_time=sim_time, anonymized=anonymize, save_levels=save_levels,
                       backgrounds_spec=S.BACKGROUNDS, n_runs=len(manifest)),
                  fh, indent=2, default=str)
    if verbose:
        print(f"wrote {len(manifest)} runs to {root}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="ds_v4")
    ap.add_argument("--seeds", type=int, default=10, help="use seeds 0..N-1")
    ap.add_argument("--backgrounds", nargs="+", default=["thin", "thick"], choices=list(S.BACKGROUNDS))
    ap.add_argument("--scenarios", nargs="+", default=None, choices=list(S.SCENARIOS))
    ap.add_argument("--sim-time", type=int, default=2000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-levels", action="store_true", help="skip observable/lob_levels.csv")
    ap.add_argument("--no-anonymize", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    generate(args.name, seeds=args.seeds, backgrounds=args.backgrounds, scenarios=args.scenarios,
             out_dir=args.out, sim_time=args.sim_time, save_levels=not args.no_levels,
             anonymize=not args.no_anonymize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
