# Detection v3: the Sethi detector on the v7 sweeps

v3 runs the same detector and scoring as v2 on the five `washtrade/v7` sweeps. In those sweeps
the wash adversary:

- rotates its wallets from a pool;
- splits each round into random-size unit trades;
- leaves small residual positions.

v3 adds no new detector. Per-run detection and evaluation is `detection.v2.experiment.detect_one`,
unchanged (configs `paper`, `no_propagation`, `theta_hi_1`; see `detection/v2/README.md`). The
v1 detector reads only `observable/trades.csv`. Prices are carried along but not scored, so the
last-digit price differences between Windows- and Linux-generated v7 data (see
`washtrade/v7/README.md`) do not change any detection result.

New in v3: **control-calibrated thresholds** (`calibrate.py`). The fixed θ = 0.9 flags a very
different share of honest volume in each background, so it is a poor basis for comparing them.
For each background and curve config, `theta_cal` is the smallest θ on the grid 0.50, 0.51, …,
1.00 whose pooled trade FPR over that background's control runs is at most 1%. If no θ reaches
1%, `theta_cal` is inf and nothing is flagged. Every run is then scored at its background's
`theta_cal`.

The control runs come from the `scenarios` sweep, scenario "control", which covers all five
backgrounds at the full seed count. The `evasive` sweep's control cells are the same runs: same
spec, same seeds.

## Files

| File | Content |
|---|---|
| `experiment.py` | Sharded, resumable runs over the v7 sweeps (`all`, `run`, `merge`, `calibrate`) |
| `calibrate.py` | Control thresholds, and every run scored at them |
| `run_smoke.py` | Smoke test, ends with `RESULT: ALL PASS`. Generates its own small v7 data |
| `results_detection_v3.ipynb` | Results: thresholds, the fixed triad vs the full adversary, one tactic at a time, the mechanism, single-factor sweeps. It only reads results; set `WASH_V7_DATA` first. `RUN_DETECTION = True` reruns the detection |

## Running

The data root is the folder holding `ds_v7_<sweep>/`. Pass it with `--data`, or set
`WASH_V7_DATA`; the default is the repository's `washtrade/v7/data`. From the repository root:

```bash
python "capstone codes/detection/v3/run_smoke.py"
python "capstone codes/detection/v3/experiment.py" all --jobs 15 --data <v7 data root>
python "capstone codes/detection/v3/experiment.py" calibrate --data <v7 data root>   # again, e.g. --fpr 0.05
```

`all` runs every sweep, merges it, then calibrates. Output goes to
`<data>/ds_v7_<sweep>/detection_v3/`:

| File | One row per |
|---|---|
| `results.csv` | (config, run): v1's metrics, same-group recall and precision, design factors `f_*` |
| `groups.csv` | (config, run, wash group): pool size, wallets in the graph, leakage, outsider score, mean scores, formula prediction |
| `curves.csv` | (curve config, run, θ): true and false positive volume, and wallet counts |
| `wallets.csv` | (run, wallet), `paper` config: position steps, closures, random-walk prediction, scores, truth |
| `calibrated.csv` | (curve config, run): trade recall, precision and FPR, same-group recall, wallet recall and FPR at `theta_cal` |

The thresholds themselves are in `<data>/detection_v3_thresholds.csv`.

## Main results (200 seeds per cell, `paper` config)

**Thresholds.** A pooled control FPR of 1% needs `theta_cal` = 0.84 (thin), 0.81 (thick),
0.99 (active_thin), 0.98 (active) and 0.97 (active_dense). At θ = 0.9 the active backgrounds
flag 42–44% of honest volume.

**Single tactics barely matter in thin and thick.** At gap 0, AP stays at 0.99–1.00 and recall at
`theta_cal` at 0.91–1.00 for each tactic alone. This includes a 96-wallet pool, where a wash
wallet returns to zero about 1.5 times per run instead of about 30. The reason is that the
initial score is binary: one closure is enough. A residual alone does not stop closures, because
wallets cross zero again in their next round.

**Combined tactics do work.** Rotation from 96 wallets, round sizes 1–4 and residuals leave about
one return to zero per wash wallet.

| Adversary (thin / thick) | Recall at `theta_cal` | AP |
|---|---|---|
| Rotation + size + residual, gap 0 | 0.74 / 0.76 | |
| + camouflage, gap 0 | 0.65 / 0.69 | |
| All four tactics, gap 20 | 0.37 / 0.34 | 0.77 / 0.59 |

**Active markets.** Detection is already weak against a fixed triad: recall at `theta_cal` is
0.45 (active) and 0.30 (active_dense). The bottleneck is the honest wallets' own closures. All four
tactics reduce AP to 0.53 / 0.38 and recall to 0.08 / 0.09.

**Camouflage is the strongest add-on.** Averaged over the other tactics, AP at gap 0 falls from
1.00 to 0.92 in thin and from 0.85 to 0.63 in active.

## Notes

In the group table, `n_wallets` is the whole pool. `n_in_graph` counts only the pool wallets
that traded: a pool wallet that was never drawn into a round has no trades, so the detector
never sees it. Wallet-level metrics, as in v1, are computed over the wallets in the graph only.
An unused pool wallet is therefore neither a hit nor a miss. `group_recall_fixed` in
`groups.csv` divides by the whole pool, so it does count unused wallets as missed.
