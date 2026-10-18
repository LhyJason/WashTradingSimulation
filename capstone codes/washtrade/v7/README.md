# Wash-trading mechanism v7: an evasive wash adversary

v7 keeps v6's backgrounds, camouflage, price rules, clearing and labels, and changes only how a
wash group uses its wallets. In v1–v6 a group is two or three fixed wallets that trade a fixed
cycle one unit at a time (A → B → C → A → …). Every few events, each wallet's position is back at
exactly zero, and the same pairs trade dozens of times. v7 adds three ways to hide these patterns.
With default arguments, v7 reproduces v6 bit for bit, and so v3 and v5 as well
(`run_smoke.py`, check A).

## The adversary

A group's `n_wallets` is now a **pool** of P wallets. The group trades in **rounds**. A round is
one closed cycle over k = `cycle_len` wallets w_1 … w_k drawn from the pool:

```text
edges     w_1 -> w_2, w_2 -> w_3, ..., w_k -> w_1      (seller -> buyer)
size      Q units on every edge
residual  the closing edge w_k -> w_1 trades Q + delta  ->  w_1 ends the round at +delta, w_k at -delta
```

| Knob | Fields | Default (= v6) |
|---|---|---|
| Rotation: which k wallets trade each round | `rotation` ("fixed", "random", "lru"), `cycle_len` | "fixed", k = P |
| Random size: Q uniform on {lo, …, hi} | `q_range`, `size_exec` ("units", "block"), `unit_lam` | Q = 1 |
| Residual: closing edge off by delta with probability p, \|delta\| ≤ d | `residual_prob`, `residual_max` | p = 0 |

Each knob draws from its own seeded random stream. As a result:

- the event-timing stream is v3's whatever the knobs (check B: rotation leaves every event time
  unchanged);
- changing one knob does not reshuffle the draws of the others.

### Design choices to review

- **Size is executed as unit trades by default.** Every ZI order has quantity 1 (hard-coded in
  `ZIAgent.take_action`). A wash leg with quantity > 1 is therefore wash on sight: no honest
  trader ever trades two units at once. With `size_exec="units"`, a round of size Q trades each
  edge as Q consecutive unit events, so every leg on the tape has quantity 1. `"block"` (one event
  of quantity Q) is kept as a contrast in the `size` sweep.
- **Residuals are bounded.** delta is flipped, or set to 0, if it would push a wallet's planned
  position outside [−d, d]. "Small residual" stays small instead of drifting like a random walk.
- **The group as a whole still nets to zero.** Its wallets only trade with each other. Residuals
  only stop individual wallets from returning exactly to zero.
- **Camouflage on a pool keeps the market's extra activity fixed.** In `evasive` and
  `evasive_camo`, every pool wallet camouflages at rate 0.06 / P. The market therefore gains the
  same ordinary activity whatever the pool size, instead of 96 extra ZI-like traders at P = 96.

## Sweeps (default 50 seeds)

| Sweep | Factors | Cells | Runs @50 | Export |
|---|---|---|---|---|
| `scenarios` | 5 backgrounds × 12 named scenarios (4 v6 baselines + 8 v7) | 60 | 3,000 | slim |
| `rotation` | 3 bg × cycle {2, 3} × gap {0, 5} × pool {k, 6, 12, 24, 48, 96} | 72 | 3,600 | tape |
| `size` | 3 bg × gap {0, 5} × (Q = 1 + hi {2, 4, 8} × {units, block}) | 42 | 2,100 | tape |
| `residual` | 3 bg × gap {0, 5} × (off + p {.1, .25, .5, 1} × d {1, 2}) | 54 | 2,700 | tape |
| `evasive` | 4 bg × gap {0, 5, 20} × pool {3, 12, 96} × Q {1, 1–4} × p {0, .5} × camouflage {off, on}, + 1 control per bg | 292 | 14,600 | tape |
| total | | 520 | 26,000 | |

- "3 bg" means thin, thick and active; `evasive` adds active_dense.
- `evasive` is the headline factorial. Its control runs are what a detector's threshold
  calibration needs. Seeds are shared across sweeps, so every sweep gets the same control runs.
- Detection reads only `observable/trades.csv`, so the tape exports are enough for it.

## Cost

- A pilot of all five sweeps at 2 seeds (1,040 runs) took 96 s with 8 parallel jobs and used
  136 MB.
- 50 seeds (26,000 runs) therefore take about 40 minutes with 8 jobs and about 3.4 GB.
- 100 seeds take about 80 minutes and 7 GB. One run costs about 0.35 s of CPU.

**A desktop is enough at 50–100 seeds.** Great Lakes pays off for several hundred seeds, many
more cells, or longer runs. One caveat for longer runs: with T > 2000, the active backgrounds'
r = 1e-3 has the same silent middle that r = 0.01 has at T = 2000. Scale r by about 2000 / T
to keep the honest market active.

## New labels and metrics

**Labels:**

- `labels/wash_rounds.csv`: one row per round, with members (wallet codes, in cycle order), size,
  residual, number of events, planned volume, and open and close time.
- `labels/wash_events.csv` gains `round_idx`, `unit_idx` and `piece`.
- `labels/wallets.csv` gains `rounds_used`. Pool wallets that were never drawn have 0 and no legs.

**Metrics** (`analysis.py`, written to `meta.json`, `manifest.csv` and `groups.csv`):

| Metric | What it measures |
|---|---|
| `n_wallets_used`, `rounds_per_used_wallet` | How widely the pool is used |
| `n_pairs`, `events_per_pair_mean` / `_max` | How often the same pair trades |
| `mean_round_size`, `mean_wash_leg_qty` | Round size, and leg quantity (above 1 only with "block") |
| `residual_round_share` | Share of rounds with a residual. Below p because of the bound |
| `zero_returns_per_used_wallet` | How often a wash wallet's net position, over all its legs, returns to exactly zero: the event a closure-based detector counts |
| `round_end_nonzero_share`, `round_end_abs_position` | Realized positions of a round's members right after the round ends |
| `honest_zero_returns_per_agent`, `honest_ever_zero_share` | The same closure count for honest agents |

The round-end positions use round ends, not the end of the run, because a run usually stops in
the middle of a round.

## Pilot observations (2 seeds per cell: direction only, not results)

**Pair repeats.** Wash events per pair fall from about 30 (fixed triad) to about 1 (pool 96).
Returns to zero per wash wallet fall from about 29 to about 1, roughly the honest level in the
active market (about 3 per agent in control runs).

**Residuals alone change little.** Returns to zero per wash wallet only fall from about 29 to
about 21 at any p. With |delta| ≤ d, a wallet's next round moves it back through zero.

**Unit-executed size helps the adversary; block size does not.** Unit-executed size lowers
returns to zero, because each round takes Q times as many events. Block size leaves them
unchanged, and it is visible on the tape anyway.

**First detector check.** The unchanged v1 detector, at θ = 0.9 and gap 0, was compared between
the fixed triad and the full evasive group (pool 96, Q 1–4, p 0.5, camouflage). On thin and thick,
recall fell from about 1.0 to about 0.5, while precision stayed above 0.9. This was a scratch run,
not part of v7. Running a detector over v7 belongs in a detection version.

## Quick start

The simplest route is the notebook `run_v7.ipynb`, set up like `run_v6.ipynb`. It runs the smoke
test, generates every sweep in parallel shards, merges and hashes each sweep, and draws
quick-look figures. Set the number of seeds and jobs in its configuration cell.

From the command line, run from the repository root:

```bash
python "capstone codes/washtrade/v7/run_smoke.py"                         # ends with RESULT: ALL PASS
python "capstone codes/washtrade/v7/generate.py" list
python "capstone codes/washtrade/v7/generate.py" run   --sweep evasive --seeds 50
python "capstone codes/washtrade/v7/generate.py" merge --sweep evasive
```

Output goes to `v7/data/ds_v7_<sweep>/`, with the same layout as v6.

## Great Lakes (Slurm)

The scripts are in `v7/slurm/` (default 50 seeds). Run every command from the repository root on
the cluster.

1. **Environment, once:** `PY_MODULE=<a python/3.12.x module> bash "capstone codes/washtrade/v7/slurm/setup_env.sh"`.
   It saves the module name in `.venv-cluster/PY_MODULE`; the job scripts load the same module.
   The environment only has to be rebuilt when `requirements-cluster.txt` changes.
2. **Check that the cluster reproduces local results.** Run the smoke test in a short job. Then
   generate the same small sweep on both machines, run `generate.py hash` on each, and compare
   the two files with `generate.py compare <local hashes.csv> <cluster hashes.csv>`.
   `hash` normalizes line endings: pandas writes CRLF on Windows and LF on Linux, so raw file
   hashes never match across the two. The v6 `hash` does not do this.
3. **Submit each sweep:**
   `ACCOUNT=<slurm account> DATA_ROOT=<scratch dir> bash "capstone codes/washtrade/v7/slurm/submit.sh" <sweep> [seeds] [n_shards]`.
   `ACCOUNT` is passed to `sbatch`, so the `CHANGE_ME` placeholder in the `.sbatch` files can stay.

**Windows and Linux are not bit-identical, but they agree on every trade.** This was checked on
2026-10-12 on the first Great Lakes run: 436 verification runs (rotation, 2 seeds; evasive, 1 seed),
plus 26,000 runs paired between the local 50-seed data and the cluster's 200-seed data.

- `compare` reports `NOT IDENTICAL`, but only float columns differ.
- Every non-float value is identical: every trade's time, wallet, side and quantity, every wash
  event, every round.
- Prices differ by at most 1.2e-4 (relative 1.3e-9), wallet cash by at most 2.4e-4.
- All summary metrics of the 26,000 paired runs are equal.

The cause is platform floating-point code, not CPU features: forcing torch's `ATEN_CPU_CAPABILITY`
to `avx2` or `avx512` locally gives the same output, and neither matches the cluster. A likely
candidate is the C math library behind torch's normal sampling (MSVC on Windows, glibc on Linux).
Treat the cluster as the reference environment. Do not mix seeds generated on the two platforms
in one dataset: prices differ in the last digits.

`requirements-cluster.txt` also pins `tqdm`. `marketsim/simulator/reward_model_builder.py`
imports it, so it is needed; v6's list lacks it.

## Notes

- `observable/trades.csv` times are one step later than the labels. The offset comes from
  upstream `market.step`, which advances its clock before clearing, and is inherited from v3.
  `analysis.round_end_positions` applies the same +1 (`CLEAR_TIME_OFFSET`).
- After an interception the group does not repair positions (as in v3). In the active markets,
  round-end positions are therefore non-zero even without residuals.
- No HBL background and no new price rule in v7.
