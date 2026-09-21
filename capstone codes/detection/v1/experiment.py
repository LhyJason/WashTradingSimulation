r"""experiment.py -- run detector configurations over a whole washtrade.v3 dataset.

Run (from the repo root):
  python "capstone codes/detection/v1/experiment.py" --dataset "capstone codes/washtrade/v3/data/ds_v3"

Writes <dataset>/detection_v1/results.csv (one row per config x run) and wallet_scores.csv /
pair_scores.csv (pooled scores for curves), unless --no-save.

Configurations (``CONFIGS``):
  paper            the paper's Algorithms 1 + 2 as written; single market -> x0 = 1{Q > 0}
  windows10        x0 over 10 equal time windows used as pseudo-markets (single-asset adaptation)
  no_propagation   ablation: skip Part II, final score = x0 (what does the network step add?)
  theta_hi_1       sensitivity: Algorithm 2 search window [0.8, 1.0] instead of [0.8, 0.99]
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_CAP)
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from detection.v1.evaluate import attach_truth, evaluate
from detection.v1.pipeline import DetectorConfig, detect
from washtrade.v3.export import load_labels, load_meta, load_observable

CONFIGS = {
    "paper":          DetectorConfig(),
    "windows10":      DetectorConfig(n_windows=10),
    "no_propagation": DetectorConfig(propagate=False),
    "theta_hi_1":     DetectorConfig(theta_hi=1.0),
}


def run_dataset(dataset_dir, configs=None, runs=None, keep_scores=True, verbose=True):
    """Detect + evaluate every (config, run). Returns (results, wallet_scores, pair_scores)."""
    configs = CONFIGS if configs is None else configs
    manifest = pd.read_csv(os.path.join(dataset_dir, "manifest.csv"))
    if runs is not None:
        manifest = manifest[manifest["run_id"].isin(runs)]

    results, wallet_rows, pair_rows = [], [], []
    t0 = time.time()
    for n, row in enumerate(manifest.itertuples(), start=1):
        rdir = os.path.join(dataset_dir, row.path)
        trades = load_observable(rdir)["trades"]
        horizon = load_meta(rdir)["config"]["sim_time"]
        labels = load_labels(rdir)
        wl = labels["wallets"].set_index("wallet")
        ids = dict(run_id=row.run_id, background=row.background, scenario=row.scenario, seed=row.seed)
        for name, cfg in configs.items():
            det = detect(trades, cfg, horizon=horizon)
            results.append(dict(config=name, **ids, **evaluate(det, labels)))
            if keep_scores:
                pairs, wallets = attach_truth(det, labels)
                w = wallets.reset_index()
                w = w.assign(config=name, **ids,
                             camouflage=w["wallet"].map(wl["camouflage"]).to_numpy(),
                             behaviour=w["wallet"].map(wl["behaviour"]).to_numpy())
                wallet_rows.append(w)
                pair_rows.append(pairs.loc[pairs["in_graph"], ["time", "buyer", "seller", "qty", "score",
                                                               "flag_fixed", "flag_star", "is_wash"]]
                                 .assign(config=name, **ids))
        if verbose and (n % 30 == 0 or n == len(manifest)):
            print(f"  {n:4d}/{len(manifest)} runs  ({time.time() - t0:5.1f}s)")

    res = pd.DataFrame(results)
    ws = pd.concat(wallet_rows, ignore_index=True) if wallet_rows else pd.DataFrame()
    ps = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    return res, ws, ps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    res, ws, ps = run_dataset(args.dataset, {k: CONFIGS[k] for k in args.configs})
    if not args.no_save:
        out = os.path.join(args.dataset, "detection_v1")
        os.makedirs(out, exist_ok=True)
        res.to_csv(os.path.join(out, "results.csv"), index=False)
        ws.to_csv(os.path.join(out, "wallet_scores.csv"), index=False)
        ps.to_csv(os.path.join(out, "pair_scores.csv"), index=False)
        print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
