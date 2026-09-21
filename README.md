# Wash Trading in a Simulated Market: Generation and Detection

A capstone project that studies wash trading as a market pathology on top of
[PyMarketSim](https://github.com/umichsrg/pymarketsim), an agent-based limit-order-book simulator.

- **Part 1 – generation (`washtrade`).** Build wash-trading agents and a controller that produce
  *ground-truth-labelled* market data: who wash-traded, in which events, and with what realized fills.
- **Part 2 – detection (`detection`).** Apply the network-based detector of Sirolly, Ma, Kanoria & Sethi (2026),
  *Network-Based Detection of Wash Trading* (position-closing scores + propagation over the counterparty graph +
  a market-specific threshold) to that data, and measure real precision / recall against the known labels.

Status: work in progress. Working defaults are marked `TODO` in the code and notebooks.

## Repository layout

```
marketsim/            PyMarketSim (upstream simulator, see "Third-party code" below)
repro/composable/     ComposableSimulator: mix agent types in one market; reused by washtrade
capstone codes/
  washtrade/          Part 1: wash-trading mechanism, versions v1 -> v3 (each version is frozen once superseded)
    v1/               active co-arrival, one directed edge per wash event
    v2/               2-wallet groups, configurable leg gap `wash_gap`, order-book recorder
    v3/               camouflage wallets, several wash groups, export schema, dataset generator
  detection/
    v1/               Part 2: closures -> propagation -> threshold, pairing, evaluation
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

# 3. Run the detector configurations over the dataset (~30 s)
python "capstone codes/detection/v1/experiment.py" --dataset "capstone codes/washtrade/v3/data/ds_v3"
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

Open them with Jupyter from anywhere inside the repository; each notebook locates the repository root itself.

## Dataset layout (produced by `washtrade/v3/generate.py`)

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
