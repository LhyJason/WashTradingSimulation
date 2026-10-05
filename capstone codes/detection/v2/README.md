# Detection v2: the Sethi detector on the v6 sweeps

v2 runs the network-based detector of Sirolly, Ma, Kanoria and Sethi (2026) on every run of the
six `washtrade/v6` sweeps (6,460 runs at 20 seeds) and evaluates it against the labels. The
detector itself is v1's, imported unchanged: pro-rata pairing, closures, score propagation, and
Algorithms 1 and 2. v1 is not modified.

## Files

| File | Content |
|---|---|
| `evaluate.py` | v1's metrics, a same-group truth label (for fills between two wash groups), the per-group table for the leakage identity, threshold curves at theta = 0.50 ... 1.00 |
| `closing.py` | random-walk return probability, and a per-wallet table (closures, random-walk prediction, scores, truth) |
| `experiment.py` | sharded, resumable runs over the v6 sweeps (`all`, `run`, `merge`) |
| `run_smoke.py` | 39 checks, ends with `RESULT: ALL PASS` |
| `results_detection_v2.ipynb` | runs all of v2 (smoke test, missing v6 data, detection) and shows the results |

## Configurations

| Config | Meaning |
|---|---|
| `paper` | Algorithm 1 with theta = 0.9; Algorithm 2 searches theta* in [0.8, 0.99] with spillover Y <= 0.1 |
| `no_propagation` | ablation: final score = initial closing score x0 |
| `theta_hi_1` | Algorithm 2 searched in [0.8, 1.0]; same scores as `paper` |

## Running

From the repository root, after the v6 datasets exist (`washtrade/v6/run_v6.ipynb`):

```bash
python "capstone codes/detection/v2/run_smoke.py"
python "capstone codes/detection/v2/experiment.py" all --jobs 8
```

Or run `results_detection_v2.ipynb`, which also generates missing v6 data. With 15 jobs, detection
over all sweeps takes about 4 minutes.

Output per sweep, in `washtrade/v6/data/ds_v6_<sweep>/detection_v2/` (git-ignored):

| File | One row per |
|---|---|
| `results.csv` | (config, run): v1's metrics, same-group recall / precision, design factors |
| `groups.csv` | (config, run, wash group): leakage, outsider score, mean initial and final score, formula prediction |
| `curves.csv` | (config, run, theta): true and false positive volume and wallet counts |
| `wallets.csv` | (run, wallet), `paper` config: position steps, closures, random-walk prediction, scores, truth |

## Definitions

- **Wash trade (truth).** A pair (buy leg, sell leg) is a wash trade iff both legs are wash orders.
  `is_wash_same_group` additionally requires both legs to come from the same group.
- **Leakage** of a wash group G is the share of G's volume traded with wallets outside G, computed
  from the detector's own pro-rata volumes V.
- **Leakage identity.** At the propagation fixed point,
  `x_bar = x0_bar + leak * (s_bar - x_out)`, where `s_bar` is the volume-weighted score of G's
  outside counterparties and `x_out` that of G's own wallets weighted by their outside volume.
  When G's wallets share one score this becomes `x_bar = (x0_bar + leak * s_bar) / (1 + leak)`.
- **Closure.** The market has no settlement or forced liquidation. A closure is a net position
  returning to 0 on the tape (the paper's `X <= c * M` with `c = 0.005`; positions here are
  integers no larger than 10).

## Main results (20 seeds)

- The leakage formula reproduces the mean score of all 5,940 wash groups (mean absolute error
  0.0006). Outsiders score about 0.48 in the r = 0.01 backgrounds and about 0.83 in the active
  ones, not 0.
- In the PyMarketSim Table-3 markets (`thin`, `thick`, r = 0.01), Algorithm 1 flags 1.5% and 0.2%
  of honest volume on control runs. A threshold calibrated on control runs (pooled FPR <= 1%) is
  0.94 and 0.82. In `thick` it raises recall at gap 20 from 0 to about 0.8.
- In the active markets (r = 1e-3), honest positions behave like random walks. The share of
  honest wallets that close matches the random-walk prediction within a few points, and
  about 80% close at least once. Algorithm 1 then flags 40–45% of honest volume on control runs.
  Algorithm 2 almost never selects a threshold, and a calibrated threshold does not recover recall.
- In the r = 0.01 markets, honest wallets close far less often than a random walk with the same
  number of trades. Their few trades follow private values in one direction, and this is what
  makes the closing signal informative there.
