# Wash-trading mechanism v6: active honest market, camouflage intensity, cluster sweeps

v6 keeps v3's wallets, cycles, timing, gap, cancellation, clearing and labels. It adds what a
detectability study needs and the tooling to run it at scale. v1–v5 are unchanged.

## What is new

| Feature | Where | Default (= v3) |
|---|---|---|
| Background mean reversion `r` as a scenario factor | `scenarios.BACKGROUNDS`, any `ComposableSimulator` keyword passes through | `r = 0.01` |
| Camouflage intensity: own arrival rate of camouflage wallets, private random streams | `WashGroup.camouflage_lam` | `None` (background rate, shared stream) |
| Partial camouflage: only the first k wallets of a group camouflage | `WashGroup.n_camouflage` | all wallets if `camouflage=True` |
| Price rule per group: v3 midpoint or v5 microprice | `WashGroup.price_rule`, `WashGroup.price` | `"mid"` |
| Background activity, camouflage share and per-group leakage metrics | `analysis.py` | – |
| Six factorial sweeps, sharded and resumable generation, merge, hashes | `design.py`, `generate.py` | – |
| Slurm job-array templates | `slurm/` | – |

With default arguments v6 reproduces v3 bit for bit; with `price_rule="microprice"` it reproduces
v5 bit for bit (`run_smoke.py`, checks A and B).

## Why `r` matters

A ZI values the asset at its forecast of the final fundamental,

```text
estimate_t = (1 - rho_t) * mean + rho_t * f_t,    rho_t = (1 - r) ** (T - t)
```

With the PyMarketSim Table-3 value `r = 0.01` and `T = 2000`, `rho_t` is close to 0 until the last
few hundred steps. After an opening burst driven by private values, every ZI values the asset at
the same constant and honest trading stops; it resumes only near the end. Adding ZIs or raising
their arrival rate makes these two bursts larger but leaves the middle of the run silent
(`run_smoke.py` check C: under 10% of honest legs fall in the middle half of the run). With
`r = 1e-3`, honest trading covers the whole run.

Means over 20 seeds (`scenarios` sweep). Honest activity is from the control runs; the wash
columns are from the triad at gap 0.

| Background | Honest legs per ZI | Honest legs in middle half | Wash share | Self-trade rate | Leakage |
|---|---|---|---|---|---|
| `thick` (30 ZI, lam 2e-2, r 1e-2) | 4.3 | 0.03 | 0.59 | 0.97 | 0.02 |
| `active` (30 ZI, lam 2e-2, r 1e-3) | 21.0 | 0.51 | 0.22 | 0.82 | 0.12 |

In the active market a gap-0 event can fail. On 10 seeds of `active`/`triad`, 180 of 943 events
failed. In all 180, a ZI order posted in the same step crossed the midpoint, and exactly one wash
leg filled: the ZI order had price priority over the other wash leg.

## Backgrounds and scenarios

Backgrounds: `thin`, `thick` (v3), `active`, `active_thin`, `active_dense` (see `scenarios.py`).
Scenarios: v3's nine, plus `triad_camo_hi`, `triad_camo_one`, `triad_micro` and `hard_hi`.

## Sweeps

| Sweep | Factors | Runs at 20 seeds | Export |
|---|---|---|---|
| `scenarios` | 5 backgrounds x 13 scenarios | 1300 | full |
| `bg_grid` | n_zi x lam x r x {control, triad} | 1440 | tape |
| `gap` | 3 backgrounds x {2, 3} wallets x 10 gaps | 1200 | slim |
| `camo` | 3 backgrounds x gap {0, 5} x (off + 6 rates x n_camouflage {1, 3}) | 1560 | slim |
| `intensity` | 3 backgrounds x gap {0, 5} x 5 wash rates | 600 | slim |
| `price` | 3 backgrounds x gap {0, 5, 20} x {mid, microprice} | 360 | slim |

Export modes: `full` = v3 layout; `slim` = without `lob_levels.csv`; `tape` = no LOB files
(trades and labels only). All six sweeps at 20 seeds (6460 runs) took 11 minutes on a desktop
with 15 parallel jobs (about 54 core-minutes in total) and take 1.6 GB of disk.

## Quick start (local)

The simplest way to run everything is the notebook `run_v6.ipynb`. It runs the smoke test,
generates every sweep in parallel shards (same commands as the cluster jobs), then merges and
hashes each sweep and draws quick-look figures. The number of seeds and parallel jobs are set in
its configuration cell. With 20 seeds and 15 jobs it took 11 minutes.

From the command line, run from the repository root:

```bash
python "capstone codes/washtrade/v6/run_smoke.py"                         # ends with RESULT: ALL PASS
python "capstone codes/washtrade/v6/generate.py" list
python "capstone codes/washtrade/v6/generate.py" run   --sweep gap --seeds 20
python "capstone codes/washtrade/v6/generate.py" merge --sweep gap
```

Output goes to `v6/data/ds_v6_<sweep>/`: `design.csv`, `runs/<run_id>/`, `manifest.csv` (one row
per run, design factors prefixed `f_`) and `groups.csv` (one row per run and wash group).

## Great Lakes (Slurm)

All commands run from the repository root on the cluster.

1. **Environment, once:** `bash "capstone codes/washtrade/v6/slurm/setup_env.sh"`. Check the
   Python module name with `module avail python` first. In `run_sweep.sbatch` and `merge.sbatch`,
   replace `CHANGE_ME` with your Slurm account.
2. **Check that the cluster reproduces local results, before any real run.**
   - Run `python "capstone codes/washtrade/v6/run_smoke.py"` on a login node or in a short job.
   - Generate the same small sweep on both machines, then run `generate.py hash --sweep price` on
     each, for example with `run --sweep price --seeds 2`.
   - Compare the two `hashes.csv` files. If they are identical, local and cluster data agree bit for
     bit. If not, treat the cluster as the reference environment and generate every reported
     dataset there.
3. **Submit a sweep:** `bash "capstone codes/washtrade/v6/slurm/submit.sh" <sweep> [seeds] [n_shards]`.
   This submits a job array of shards and a merge job that runs after it. Set `DATA_ROOT` to
   write to scratch storage instead of the repository.
4. **Re-submitting is safe.** Finished runs (those with `meta.json`) are skipped, failed runs leave
   a traceback in `errors/` and are retried, and each run is written to `<run>.tmp` and renamed
   when complete. On Windows the rename is retried, because a virus scanner or search indexer
   can briefly lock the files that were just written.

## Notes

- `observable/trades.csv` times are still one step later than `lob.csv` and the labels. The offset
  comes from upstream `MatchedOrder` and is inherited from v3. Joins on `leg_id` are unaffected.
- `detection/v1` loads runs with v3's `load_observable`, which requires `lob.csv`. It therefore
  reads `full` and `slim` exports, but not `tape` exports. Use `washtrade.v6.export.load_observable`
  for those.
- Leakage (`analysis.group_leakage`) is the share of a group's volume whose counterparty is not
  in the group. It is computed per clear under pro-rata pairing (what the network detector sees)
  and under max-match pairing.
- An HBL background is not part of v6.
