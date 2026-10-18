# Wash Trading in a Simulated Market: Generation and Detection

A capstone project that studies wash trading as a market pathology on top of
[PyMarketSim](https://github.com/umichsrg/pymarketsim), an agent-based limit-order-book simulator.

- **Part 1 – generation (`washtrade`).** Build wash-trading agents and a controller that produce
  *ground-truth-labelled* market data: who wash-traded, in which events, and with what realized fills.
- **Part 2 – detection (`detection`).** Apply the network-based detector of Sirolly, Ma, Kanoria & Sethi (2026),
  *Network-Based Detection of Wash Trading* (position-closing scores + propagation over the counterparty graph +
  a market-specific threshold) to that data, and measure real precision / recall against the known labels.

Status: work in progress. Working defaults are marked `TODO` in the code and notebooks.
Collaboration rules (branches, Pull Requests, versioning): see [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository layout

```
marketsim/            PyMarketSim (upstream simulator, see "Third-party code" below)
repro/composable/     ComposableSimulator: mix agent types in one market; reused by washtrade
capstone codes/
  washtrade/          Part 1: wash-trading mechanism, versions v1 -> v7 (v1-v7 are frozen; new work goes into v8, ...)
    v1/               active co-arrival, one directed edge per wash event
    v2/               2-wallet groups, configurable leg gap `wash_gap`, order-book recorder
    v3/               camouflage wallets, several wash groups, export schema, dataset generator
    v4/               v3 + random limit prices around the midpoint (buy/sell limits cross inside the spread); own export + generator
    v5/               v3 + one common random price per event, centred on the microprice; volatility-capped radius; parameter-sweep CSV
    v6/               v3 + active honest market (mean reversion r), camouflage intensity, six sweeps, sharded generator, Slurm templates
    v7/               v6 + evasive adversary: wallet-pool rotation, random round size executed as unit trades, bounded residuals; five sweeps
  detection/
    v1/               Part 2: closures -> propagation -> threshold, pairing, evaluation (v1-v3 are frozen; new work goes into v4, ...)
    v2/               the v1 detector on the v6 sweeps: leakage identity, random-walk benchmark for closures, calibrated thresholds
    v3/               the v2 detector on the v7 sweeps; thresholds calibrated per background to a 1% control FPR
```

The folder name `capstone codes` contains a space, so it is not directly importable. Every entry point and
notebook adds both the repository root and `capstone codes/` to `sys.path` and then imports
`washtrade.v3`, `detection.v1`, etc.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code was developed and verified with Python 3.12 and the versions in `requirements.txt`.
No `pip install -e .` is needed.

## Quick start

Run everything from the repository root.

```bash
# 1. Smoke tests (each prints ALL PASS when the checks succeed)
python "capstone codes/washtrade/v1/run_smoke.py"
python "capstone codes/washtrade/v2/run_smoke.py"
python "capstone codes/washtrade/v3/run_smoke.py"
python "capstone codes/detection/v1/run_smoke.py"

# 2. Generate the labelled dataset: 2 backgrounds x 9 scenarios x 10 seeds = 180 runs (~30 s)
python "capstone codes/washtrade/v3/generate.py" --name ds_v3 --seeds 10

# 2b. (optional) v4/v5 smoke tests and a v4 dataset with randomized wash prices (schema washtrade.v4/1)
python "capstone codes/washtrade/v4/run_smoke.py"
python "capstone codes/washtrade/v4/generate.py" --name ds_v4 --seeds 10
python "capstone codes/washtrade/v5/run_smoke.py"      # v5 has no export/generator of its own yet

# 2c. (optional) v6: smoke test, then every sweep (6,460 runs at 20 seeds); or run v6/run_v6.ipynb
python "capstone codes/washtrade/v6/run_smoke.py"
python "capstone codes/washtrade/v6/generate.py" list
python "capstone codes/washtrade/v6/generate.py" run   --sweep gap --seeds 20
python "capstone codes/washtrade/v6/generate.py" merge --sweep gap

# 2d. (optional) v7: smoke test, then the evasive sweep (14,600 runs at 50 seeds); or run v7/run_v7.ipynb
python "capstone codes/washtrade/v7/run_smoke.py"
python "capstone codes/washtrade/v7/generate.py" run   --sweep evasive --seeds 50
python "capstone codes/washtrade/v7/generate.py" merge --sweep evasive

# 3. Run the detector configurations over the dataset (~30 s)
python "capstone codes/detection/v1/experiment.py" --dataset "capstone codes/washtrade/v3/data/ds_v3"

# 3b. (optional) detection v2 on the v6 sweeps (needs step 2c); or run detection/v2/results_detection_v2.ipynb
python "capstone codes/detection/v2/run_smoke.py"
python "capstone codes/detection/v2/experiment.py" all --jobs 8

# 3c. (optional) detection v3 on the v7 sweeps (needs all five v7 sweeps); or run detection/v3/results_detection_v3.ipynb
python "capstone codes/detection/v3/run_smoke.py"
python "capstone codes/detection/v3/experiment.py" all --jobs 8 --data "capstone codes/washtrade/v7/data"
```

Generated data is written to `capstone codes/**/data/`, which is git-ignored. Given the same seeds and the pinned
`numpy` / `torch` versions the dataset is reproduced bit for bit.

## Result notebooks

| Notebook | Content |
|---|---|
| `capstone codes/washtrade/v1/results_v1.ipynb` | v1 mechanism: net position returns to zero every cycle, wash-volume share, why `wash_eps` is not a spillover knob |
| `capstone codes/washtrade/v2/results_v2.ipynb` | 2 vs 3 wallets, the `wash_gap` scan (self-trade vs intercept rate), order-book recording |
| `capstone codes/washtrade/v3/results_v3.ipynb` | camouflage wallets, multiple groups, the `observable/` vs `labels/` split, dataset overview (needs step 2 above) |
| `capstone codes/detection/v1/results_detection_v1.ipynb` | detector evaluation on the v3 dataset (needs step 2; runs step 3 itself if its results are missing) |
| `capstone codes/detection/v2/results_detection_v2.ipynb` | the detector on every v6 sweep: false positives on honest-only markets, leakage identity, random-walk benchmark, calibrated thresholds (generates missing v6 data itself) |
| `capstone codes/detection/v3/results_detection_v3.ipynb` | the detector against the v7 adversary: calibrated thresholds, fixed triad vs full adversary, one tactic at a time (reads detection v3 results; set `WASH_V7_DATA`) |
| `capstone codes/washtrade/v6/run_v6.ipynb` | runs the whole of v6 (smoke test, all six sweeps in parallel, merge, hashes) and draws quick-look figures |
| `capstone codes/washtrade/v7/run_v7.ipynb` | runs the whole of v7 (smoke test, all five sweeps in parallel, merge, hashes) and draws quick-look figures |

Open them with Jupyter from anywhere inside the repository; each notebook locates the repository root itself.

## Wash-price randomization (v4 / v5)

v3 posts both wash legs exactly at the midpoint, which is an easy fingerprint. v4 and v5 keep v3's wallets, cycles,
timing, camouflage and labels and only change the price rule; see `PRICE_RANDOMIZATION.md` for the method, the
27-configuration sweep (2,430 runs) and the headline numbers, and `capstone codes/washtrade/v4/README.md` and
`v5/README.md` for each version's parameters.

| Version | Price rule | Exporter / generator |
|---|---|---|
| v3 | both legs at the midpoint | `v3/export.py`, `v3/generate.py` |
| v4 | sell limit ~ truncated normal around the midpoint, buy limit = sell limit + random crossing width | `v4/export.py`, `v4/generate.py` (schema `washtrade.v4/1`) |
| v5 | one common price per event, truncated normal around the microprice, tick-rounded | reuses v3 simulator; no own exporter yet |

TODO: v5 has no dataset generator, and the detector has not been run on a v4/v5 dataset in this repository, so the
detector F1 quoted in `PRICE_RANDOMIZATION.md` is not yet reproducible from the code here.

## Active background and camouflage intensity (v6)

With PyMarketSim's Table-3 mean reversion `r = 0.01` and `T = 2000`, a ZI's forecast of the final fundamental is
almost constant for most of the run, so honest trading stops after an opening burst and only resumes near the end;
more ZIs or faster arrivals do not change this. v6 makes `r` a background factor (`r = 1e-3` keeps the honest market
trading throughout), gives camouflage wallets their own arrival rate, and measures each wash group's leakage
(the share of its volume traded with outsiders). With default arguments v6 reproduces v3 bit for bit. See
`capstone codes/washtrade/v6/README.md` for the backgrounds, the six sweeps and how to run them on a Slurm cluster.

## An evasive wash adversary (v7)

In v1-v6 a wash group is two or three fixed wallets that trade one unit at a time in a fixed cycle, so every few
events each wallet's position is back at zero and the same pairs trade dozens of times. v7 keeps v6's backgrounds,
camouflage, price rules and labels and lets a group hide these patterns: it trades in rounds over wallets drawn from
a pool, each round has a random size executed as unit trades (every honest ZI order is one unit), and the closing
edge can leave a small bounded residual. With default arguments v7 reproduces v6 bit for bit. See
`capstone codes/washtrade/v7/README.md` for the knobs, the five sweeps, the new labels and metrics, and the
Windows-vs-Linux reproducibility check on a Slurm cluster.

## Detection on an active market (detection v2)

detection v2 runs the v1 detector, unchanged, on all 6,460 v6 runs. In PyMarketSim's Table-3 market
(`thin`, `thick`) it works as in v1. In the active backgrounds, honest positions move like random walks
and most honest wallets return to a zero position at least once, so the closing signal no longer separates
them. Algorithm 1 then flags 40-45% of honest volume on honest-only runs. See
`capstone codes/detection/v2/README.md` for the leakage identity, the random-walk benchmark and
control-calibrated thresholds.

## Detection against the evasive adversary (detection v3)

detection v3 runs the v2 detector, unchanged, on the five v7 sweeps, and scores every run at a threshold calibrated
per background so that at most 1% of honest volume in that background's control runs is flagged. In `thin` and
`thick`, each tactic alone barely matters: the initial score is binary, and one return to zero is enough. Combined
tactics do work: rotation from 96 wallets, random size and residuals bring recall at the calibrated threshold to
about 0.75, and about 0.35 with camouflage and a leg gap of 20. In the active markets detection is weak even against
a fixed triad (recall 0.45 / 0.30), because honest wallets close their positions too. See
`capstone codes/detection/v3/README.md`.

## Dataset layout (produced by `washtrade/v3/generate.py`; v4 writes the same layout under schema `washtrade.v4/1`, v6 under `washtrade.v6/1`, v7 under `washtrade.v7/1`)

```
<dataset>/runs/<background>__<scenario>__s<seed>/
  meta.json
  observable/   trades.csv  lob.csv  lob_levels.csv     <- all a detector may read
  labels/       wallets.csv  trade_legs.csv  wash_events.csv  wash_fills.csv  wash_orders.csv  lob_latent.csv
```

Wallet ids are anonymized, order ids are dropped, and legs are shuffled within each clear, so that nothing in
`observable/` gives the wash wallets away. See `capstone codes/washtrade/v3/export.py`.

## Third-party code

`marketsim/` is a copy of [umichsrg/pymarketsim](https://github.com/umichsrg/pymarketsim)
(Strategic Reasoning Group, MIT license, see `LICENSE.txt`). The capstone code never edits it; it only imports and
monkey-patches it. The copy carries six small compatibility patches so that it runs on current Python, gymnasium and SciPy:

1. removed a dead import of a non-existent `sampled_arrival_simulator_custom`;
2. added a missing `from typing import List`;
3. replaced the `fastcubicspline` dependency (its C extension does not build on Python >= 3.11) with an equivalent
   NumPy/SciPy implementation in `marketsim/agent/_spline.py`;
4. unified bare `from fourheap...` imports to the `marketsim.` prefix;
5. fixed a `spaces.Box` shape given as an int instead of a tuple;
6. fixed a `ZIAgent.take_action` call that no longer matches its signature.

The detector implements the algorithms of Sirolly, Ma, Kanoria & Sethi (2026); all credit for the method is theirs.
