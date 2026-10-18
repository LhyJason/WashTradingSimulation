# Detection v4: a multi-channel Sethi detector

Detection v3 ran Sethi et al.'s detector on the v7 sweeps and found that its Part I score breaks in
two ways:

- **Active markets.** Honest positions behave like random walks and return to zero about as
  often as wash positions do. At a 1% control FPR, recall against a fixed triad was only 0.45
  (active) and 0.30 (active_dense).
- **Rotating pool.** When the adversary rotates through a large wallet pool and adds residuals,
  each wash wallet closes only about once, so the binary "ever closed" score loses its contrast.

v4 keeps the network step (Part II) unchanged and adds two kinds of Part I evidence that look at
short time windows instead of the whole run. It then fuses all channels into one decision whose
false-positive rate is calibrated on control runs.

## The detector

Each channel gives every wallet an initial score $x^0 \in [0, 1]$, a share of its volume. The
score is propagated with v1's Algorithm 1 ($x = \tfrac12 x^0 + \tfrac12 Bx$), and a pair is scored
$\min(x_i, x_j)$.

| Channel | $x^0$ of a wallet | Why wash rounds score high |
|---|---|---|
| `closure` | 1 if the position ever returns to 0 (v1, unchanged = v3) | the round closes positions |
| `roundtrip_W`, W = 50, 100, 200 | share of its units undone (FIFO) by an opposite unit at most W steps later | a round undoes its positions within the round; an honest trader only over its long arrival gaps |
| `repeat_W`, W = 20, 50, 200 | share of its volume on ordered pairs (seller → buyer) that trade again within W steps | a round of size Q puts Q consecutive unit trades on each edge |

**Fusion.** For each background, $u_k(s)$ is the share of honest pair volume in the 200 control
runs whose channel-$k$ score is at least $s$, a volume-weighted tail probability. A channel only
counts once its score reaches the floor $f = 0.5$:

$$u^f_k(s) = u_k(s) \text{ if } s \ge f, \text{ else } 1, \qquad z_S = 1 - \min_{k \in S} u^f_k(\text{score}_k).$$

A pair is as suspicious as its most extreme channel. Reported methods:

- every channel alone (with the floor);
- `closure+roundtrip`, `closure+repeat`, and `v4` (all seven channels);
- two ablations without the network step: `roundtrip_100_noprop` and `repeat_50_noprop`.

**Threshold.** For each background, method and floor, θ is the smallest threshold whose pooled
control FPR is at most 1%. It is computed exactly from the control scores; v3 used a 0.01 grid. If
even the highest control score breaks the target, θ sits 1e-9 above it. The same is done at 0.1%
and 5% for the main floor.

**Why there is a floor.** It was added after the first full run of the `scenarios` sweep. The
repeat channels are exactly 0 on almost all honest control volume, so their calibrated
thresholds sat just above 0 (around 0.01). In a wash run, the network step gives every wallet
connected to the wash group a small positive score. With such low thresholds, up to 95% of honest
volume in wash runs was flagged, while the control FPR stayed at 1%. Scores are network-smoothed
volume shares, so the floor 0.5 reads "at least half of both wallets' volume looks wash-like".
This is the same kind of absolute level as Sethi's θ = 0.9. Floors 0, 0.3 and 0.7 are reported as
a sensitivity check.

The floor does not change the closure channel, whose calibrated θ is at least 0.81 everywhere. AP
is threshold-free and is computed on the floor-0 scores.

**What is not controlled.** The 1% FPR holds on control runs by construction. In wash runs,
honest wallets that trade with wash wallets (camouflage fills, interceptions) pick up their scores
through the network step. This spillover affects every method, v3 included. Every table therefore
reports the FPR on the honest volume of the wash runs next to recall.

## How the channels were chosen

The candidates were screened on a sample of the `evasive` sweep: seeds 0–29 for wash cells and
0–99 for controls, in 4 backgrounds. Kept: round trips and same-pair repeats, each at a short,
middle and long window.

Dropped:

- **Closure scored against a random-walk null** (mid-p of the number of returns to zero given
  the number of steps). It was worse than plain closure, because rotated wash wallets have too few
  steps for any test.
- **Short-window cycle decomposition**, in the style of Victor and Weintraud (2021): find directed
  cycles of trades within a window. Honest flows in active markets also contain many cycles.

The windows were fixed before the full run, and the full run uses all 200 seeds. The repeat and
round-trip windows are tied to the adversary's event rate (λ_wash = 0.05, about one wash event
every 20 steps). A slower adversary needs longer windows.

## Files

| File | Content |
|---|---|
| `channels.py` | Part I channels: `roundtrip_share`, `repeat_flags` / `repeat_scores`, `channel_scores` |
| `detector.py` | `detect_v4`: pairing and graph (v1), every channel propagated, pair scores, `_noprop` ablations |
| `fusion.py` | Control null (`Null`, `Survival`), floor, fused score, exact thresholds |
| `evaluate.py` | Per-run, per-method scoring: counts at the thresholds, AP, ROC-AUC; wallet table |
| `experiment.py` | Pass 1 `null` (control runs), pass 2 `run`/`merge` (sharded, resumable), `all` |
| `run_smoke.py` | Smoke test (closure = v1, channels on hand-made tapes, propagation invariant, fusion and thresholds, runner); ends with `RESULT: ALL PASS` |
| `results_detection_v4.ipynb` | Results. It only reads results; set `WASH_V7_DATA` first. `RUN_DETECTION = True` reruns everything |

## Running

The data root is the folder holding `ds_v7_<sweep>/`. Pass it with `--data`, or set
`WASH_V7_DATA`. From the repository root:

```bash
python "capstone codes/detection/v4/run_smoke.py"
python "capstone codes/detection/v4/experiment.py" all --jobs 15 --data <v7 data root>
```

`all` builds the null from the `scenarios` sweep's control runs (pass 1, about 30 s). It then
runs, merges and saves each sweep (pass 2). The whole run takes about 45 minutes with 15 jobs for
104,000 runs.

Outputs:

| File | One row per |
|---|---|
| `<data>/detection_v4_null.csv` | control pair: qty and every single-channel score |
| `<data>/detection_v4_thresholds.csv` | (background, method, floor): θ and pooled control FPR per target |
| `ds_v7_<sweep>/detection_v4/results.csv` | (run, method). At the main floor and the 1% target: `trade_tp/fp/pos/neg`, `trade_tp_same`, `wallet_tp/fp/pos/neg` and their ratios. Other targets: `trade_tp_0.1`, `trade_fp_5`, … Other floors: `trade_tp_f0.3`, `trade_fp_f0`, … Also `trade_ap`, `trade_roc_auc` and the design factors `f_*` |
| `ds_v7_scenarios/detection_v4/wallets.csv` | (run, wallet), `scenarios` only: every channel's `x0_*` and `x_*`, volume, degree, truth |

Pool the ratios by summing the counts over runs (volume-weighted), as the notebook does.

## Main results (200 seeds per cell; pooled control FPR 1%, floor 0.5)

`closure` is v3's detector, rescored with v4's exact thresholds; its calibrated θ equals v3's
up to the 0.01 grid. The tables pool every wash cell of the `evasive` sweep: 288 cells × 200
seeds per background, volume-weighted.

| Background | Recall, v3 → v4 | Precision, v3 → v4 | FPR in wash runs, v3 → v4 | AP, v3 → v4 |
|---|---|---|---|---|
| thin | 0.84 → 0.97 | 0.92 → 0.92 | 16.4% → 20.6% | 0.91 → 0.93 |
| thick | 0.83 → 0.93 | 0.86 → 0.88 | 9.2% → 8.5% | 0.84 → 0.90 |
| active | 0.19 → 0.70 | 0.59 → 0.74 | 1.4% → 2.6% | 0.40 → 0.61 |
| active_dense | 0.22 → 0.65 | 0.37 → 0.60 | 1.6% → 1.8% | 0.29 → 0.55 |

**Each tactic alone no longer works (gap 0, no camouflage).** Recall, v3 → v4:

| Adversary | thin | thick | active | active_dense |
|---|---|---|---|---|
| Fixed triad | 0.97 → 1.00 | 0.97 → 1.00 | 0.46 → 1.00 | 0.52 → 1.00 |
| Pool 96 | 0.99 → 0.99 | 0.98 → 0.99 | 0.54 → 0.95 | 0.53 → 0.91 |
| Q 1..4 | 0.93 → 1.00 | 0.91 → 1.00 | 0.49 → 1.00 | 0.48 → 1.00 |
| Residual 0.5 | 0.98 → 1.00 | 0.98 → 1.00 | 0.49 → 1.00 | 0.52 → 1.00 |
| Pool 96 + Q + residual | 0.74 → 0.99 | 0.76 → 0.98 | 0.31 → 0.93 | 0.27 → 0.79 |

**The two new channels cover each other.** The size tactic backfires: a round's Q units are
executed one at a time on the same edge, which leaves a burst of same-pair trades for the repeat
channel. Size-1 rounds leave no burst, but the round-trip channel still catches them. With all
four tactics at gap 0 in active, recall is 0.08 for closure, 0.21 for round trips, 0.46 for
repeats and 0.65 for v4.

**The gap still defeats detection in active markets.** Against all four tactics, recall is
0.08 / 0.06 (active / dense) at gap 5 and 0.01 / 0.02 at gap 20. A fixed triad at gap 5 is now
caught (0.06 → 1.00 active, 0.03 → 0.95 dense). In thin and thick, all four tactics at gap 20
give 0.38 → 0.71 and 0.36 → 0.39.

**The cost is spillover in thin markets.** The FPR on the honest volume of wash runs rises from
16.4% to 20.6%. It was already high for closure: honest volume in thin markets is small, and much
of it touches wash or camouflage wallets. The control FPR is 1% for every method by construction.

**Floor sensitivity** (v4, all wash cells, recall / FPR in wash runs):

| Floor | active | active_dense | thin |
|---|---|---|---|
| 0 | 0.85 / 4.4% | 0.84 / 10.9% | 0.98 / 30.1% |
| 0.3 | 0.79 / 2.8% | 0.73 / 1.8% | 0.98 / 22.3% |
| **0.5** (main) | 0.70 / 2.6% | 0.65 / 1.8% | 0.97 / 20.6% |
| 0.7 | 0.59 / 2.2% | 0.55 / 1.7% | 0.92 / 15.9% |

Floor 0.3 looks like the better operating point in active markets. The main floor stays at 0.5
because it was fixed before these results were seen.

**The network step still matters.** Without propagation, round-trip recall in active falls from
0.58 to 0.47, and AP falls for both new channels. For repeats, recall without propagation is
higher but AP is lower.

**Robust to the screening seeds.** The channel windows were screened on seeds 0–29. On seeds
30–199 alone, every pooled number in the first table changes by at most 0.01.

## Limitations and next steps

- **The repeat channel exploits how v7 executes size.** v7 runs a round's units consecutively on
  one edge, about 20 steps apart. An adversary that interleaves units across edges, or spaces
  them out, would blunt it. This is a natural v8 adversary test.
- **The windows assume λ_wash = 0.05.** A slower adversary needs longer windows, and in active
  markets longer windows admit more honest round trips.
- **Gap is the open problem in active markets.** At gap ≥ 5, a wash event's two legs are far
  enough apart for honest traders to fill them (interception), and the rounds stretch out.
  Neither new channel recovers them.
- **Spillover FPR is not controlled.** Precision in thin markets, about 0.92, is the honest
  measure there; control-run calibration cannot see spillover.
