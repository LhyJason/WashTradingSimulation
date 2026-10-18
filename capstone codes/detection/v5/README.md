# Detection v5: detection v4's detector against the v8 adversary

v5 adds no new detector. It runs detection v4 unchanged on the two `washtrade/v8` sweeps (200
seeds, 28,000 runs):

- v4's channels: closure, round trip (W = 50, 100, 200) and same-pair repeat (W = 20, 50, 200);
- each channel propagated over the counterparty graph;
- control-null fusion, floor 0.5, and thresholds at a pooled control FPR of 1%.

The v8 sweeps ask two questions:

- **`adversary`.** Can v7's evasive group escape v4's short-window channels? It can interleave a
  round's edges, keep D rounds open at once (a pipeline, so each wallet holds a position longer
  while the group stays flat), or trade at a quarter of the rate. What does each option cost it?
- **`honest`.** What changes when honest traders post orders of 1..4 units (`<bg>_mu4`)?

Thresholds are recalibrated on each of the eight v8 backgrounds' 200 control runs, from the
`honest` sweep. For the four standard backgrounds those runs are the v7 control runs bit for
bit, so the thresholds equal detection v4's (`run_smoke.py`, check B).

## Files

| File | Content |
|---|---|
| `experiment.py` | Pass 1 `null` (honest sweep's controls), pass 2 `run`/`merge` over the v8 sweeps, `all`; per-run work is v4's `detect_one` |
| `run_smoke.py` | Null over 8 backgrounds, thresholds equal to v4's on the standard backgrounds, runner; ends with `RESULT: ALL PASS` |
| `results_detection_v5.ipynb` | Results. It only reads results; set `WASH_V8_DATA` |

Run from the repository root. The full run takes about 12 minutes with 15 jobs.

```bash
python "capstone codes/detection/v5/run_smoke.py"
python "capstone codes/detection/v5/experiment.py" all --jobs 15 --data <v8 data root>
```

Outputs have the same tables as v4, in `<data>/ds_v8_<sweep>/detection_v5/`: `results.csv`, plus
`wallets.csv` for `honest`. The null and thresholds are in `<data>/detection_v5_{null,thresholds}.csv`.

## Main results (200 seeds per cell; pooled control FPR 1%, floor 0.5)

**Recall of v4 at gap 0** (`adversary` sweep). The first row is v7's execution, the "all four
tactics" adversary of detection v4. FPR in the wash runs is at most 0.09 in every cell
(thin, rate 0.05), and 0.00–0.01 in the active markets.

| Execution | lam_wash | thin | thick | active | active_dense |
|---|---|---|---|---|---|
| sequential, D = 1 (v7) | 0.05 | 0.98 | 0.95 | 0.65 | 0.42 |
| interleaved, D = 1 | 0.05 | 0.98 | 0.97 | 0.84 | 0.74 |
| sequential, D = 4 | 0.05 | 0.93 | 0.90 | 0.47 | 0.23 |
| interleaved, D = 4 | 0.05 | 0.94 | 0.89 | 0.17 | 0.20 |
| sequential, D = 8 | 0.05 | 0.74 | 0.71 | 0.20 | 0.11 |
| interleaved, D = 8 | 0.05 | 0.81 | 0.80 | 0.07 | 0.16 |
| sequential, D = 1 | 0.0125 | 0.91 | 0.87 | 0.43 | 0.24 |
| sequential, D = 8 | 0.0125 | 0.08 | 0.08 | 0.01 | 0.03 |
| interleaved, D = 8 | 0.0125 | 0.37 | 0.45 | 0.04 | 0.11 |

1. **Interleaving alone backfires.** It removes most same-pair bursts: the share of wash events
   repeated within 50 steps falls from 0.85 to 0.48. But each wallet then undoes its position
   after a single pass of the cycle, and the round-trip channel catches that. Active recall rises
   from 0.65 to 0.84.
2. **A pipeline evades v4 in the active markets.** Each wallet holds longer than every round-trip
   window, and pairs stop repeating. With D = 8 and interleaved edges, recall is 0.07 (active) and
   0.16 (active_dense). Adding a slow rate brings even thin and thick down to 0.08.
3. **The pipeline is cheap.** Wash volume is unchanged: about 46 units per 1000 steps at rate
   0.05, against 12 at rate 0.0125. The costs are elsewhere:
   - each wallet holds a unit for 310–620 steps instead of 150–190 (D = 8);
   - 45% of rounds are unfinished at the end of the run;
   - 17–23% of a wash wallet's units are open at the end, against 14% for v7's execution.

   The group as a whole stays flat. The only price is holding risk at wallet level, which the
   group hedges internally.
4. **Multi-unit honest orders weaken closure, not the new channels.** In active_mu4 and
   active_dense_mu4, the closure channel cannot reach a 1% control FPR and flags nothing.
   - v4 still catches the fixed triad (1.00 / 1.00).
   - Against block-executed rounds, whose multi-unit legs now blend in, v4 reaches 0.68 / 0.59.
     The round-trip channel catches them.
   - Against v7's execution, v4's recall falls from 0.65 to 0.40 (active) and from 0.42 to 0.22
     (active_dense).
   - FPR in wash runs stays at 0.00–0.02 in the active markets.

## What this means for detection

v4's gain came from per-wallet, short-window evidence. v8 shows that an adversary can remove
that evidence cheaply: it only has to overlap enough rounds that each wallet holds a position
longer than the windows. What it cannot hide is the group itself. Its wallets trade (almost) only
with each other, and their summed position stays flat while each one holds. Evidence measured on
clusters of the counterparty graph is therefore the natural next detector: for example, closure or
round trips of a cluster's aggregate position instead of a single wallet's. Gap remains the other
open problem (see detection v4).
