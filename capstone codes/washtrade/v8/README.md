# Wash-trading mechanism v8: an adversary that hides from short windows, and multi-unit honest orders

Detection v4 caught v7's evasive adversary with two short-window channels. v8 asks two
questions:

- what a wash group can do against those channels, and what that costs it;
- whether the channels only worked because every honest order in the simulator is one unit.

With default arguments, v8 reproduces v7 bit for bit, and so v6 and v3 as well (`run_smoke.py`,
check A).

## The two v4 channels and how v7 fed them

| v4 channel | What it reads | What v7 did |
|---|---|---|
| `repeat_W` | the same ordered pair (seller → buyer) trading again within W steps | a round of size Q traded all Q units of an edge back to back |
| `roundtrip_W` | a wallet undoing its position within W steps | one round at a time: a wallet's two legs in a round were about Q events apart |

## New adversary knobs (`WashGroup`)

Both knobs change only the *order* in which a round's unit events run. Rotation, size, residual,
event timing and every random stream are v7's, so event times do not depend on them
(checks C and D).

| Knob | Values | Effect |
|---|---|---|
| `edge_order` | `"sequential"` (v7), `"interleaved"` | Interleaved runs a round's edges round robin: unit 1 of every edge, then unit 2, and so on. A pair trades once every k events instead of Q times in a row |
| `pipeline` = D | 1 (v7), 4, 8 | D rounds are open at once and their events are taken in turn. A wallet joins a new round only once its legs in the open rounds are done, which needs D·k ≤ pool size. A wallet holds each position about D times longer; its counterparties in the round hold the opposite position, so the group stays flat |
| `lam_wash` (v3 knob) | 0.05, 0.0125 | A slower adversary: positions are held 4× longer, at a quarter of the wash volume |

**What evasion costs.** A wash group wants volume without exposure. Holding longer means carrying
price risk longer at wallet level (`hold_mean`). With a deep pipeline, many rounds are still open
when the run ends (`rounds_completed_share`, `open_unit_share`). A slower rate buys less volume
(`wash_volume_rate`). These are recorded per group, so detection can be read against the cost.

## Multi-unit honest orders (`order_size_max` = m)

`agents.MultiUnitZI` (registered as agent type `"MZI"`) is ZI with one change. On each arrival it
draws an order size k uniformly from {1..m}, from its own seeded stream, and prices the order at
the mean of the k marginal private values it walks through:

```text
BUY : estimate + mean_{i<k} pv(position + i, BUY)  - offset
SELL: estimate + mean_{i<k} pv(position - i, SELL) + offset
```

Everything else is the same as ZI: the side coin, the valuation offset, the order-id draw and the
eta rule all use the global `random` stream exactly as ZI does. With k = 1 the formula is ZI's,
and m = 1 reproduces v7 bit for bit (check B). The four-heap book already handles multi-unit
orders and partial fills.

A simulator built with `order_size_max = m` turns every `"ZI"` of the background into `"MZI"`.
Camouflage wallets become `"MZI"` too, so they keep looking like the background. The backgrounds
`<bg>_mu4` (thin, thick, active, active_dense) use m = 4, the same range as the wash round size
Q in 1..4. A block-executed wash order of Q units therefore no longer stands out by its size.

Upstream `ZIAgent` was not modified: `marketsim/` is untouched, and the new type is added to the
registry from `v8/agents.py`.

## Sweeps (200 seeds)

| Sweep | Factors | Cells | Runs |
|---|---|---|---|
| `adversary` | 4 bg × gap {0, 5} × edge order {sequential, interleaved} × pipeline {1, 4, 8} × lam_wash {0.05, 0.0125}, plus 1 control per bg | 100 | 20,000 |
| `honest` | 4 bg × honest order size {1, 1..4} × {control, fixed triad, evasive, evasive block, evasive interleaved + pipeline 8}, gap 0 | 40 | 8,000 |

- "4 bg" means thin, thick, active and active_dense.
- The wash group in both sweeps is v7's full evasive adversary: pool 96, cycles of 3, Q in 1..4,
  residual on half of the rounds, and camouflage at a total rate of 0.06.
- The `honest` sweep's control runs cover all eight backgrounds and are what a detector's
  calibration needs. For the four standard backgrounds they are the v7 control runs bit for bit.

## Metrics added (`analysis.py`, in `meta.json`, `manifest.csv`, `groups.csv`)

| Metric | Meaning |
|---|---|
| `edge_order`, `pipeline`, `max_open_rounds` | execution settings, and the most rounds open at once |
| `rounds_completed_share` | share of started rounds whose events all ran before the run ended |
| `repeat_share_50` | share of wash events whose ordered pair has another wash event within 50 steps (ground truth for v4's `repeat_50`) |
| `hold_mean`, `hold_median` | steps a wash wallet's unit position stays open before an opposite unit undoes it: all its legs, FIFO, as v4's round-trip channel matches them |
| `open_unit_share` | share of a wash wallet's units still open at the end of the run |
| `wash_volume_rate` | wash volume per 1000 steps |
| `honest_mean_order_size`, `honest_mean_leg_qty` | drawn honest order sizes (1 for ZI), and mean filled honest leg on the tape |

## Pilot (2 seeds per cell; direction only)

| Execution | lam_wash | Repeat share (50 steps) | Hold (steps) | Open units at end | Rounds completed |
|---|---|---|---|---|---|
| sequential, D = 1 (v7) | 0.05 | 0.82 | 208 | 0.17 | 0.93 |
| interleaved, D = 1 | 0.05 | 0.33 | 176 | 0.15 | 0.93 |
| sequential, D = 8 | 0.05 | 0.00 | 647 | 0.29 | 0.48 |
| interleaved, D = 8 | 0.05 | 0.00 | 369 | 0.20 | 0.48 |
| sequential, D = 1 | 0.0125 | 0.53 | 396 | 0.14 | 0.83 |

- Interleaving alone removes most same-pair repeats but shortens holding; a pipeline removes
  repeats and lengthens holding.
- The price is paid in unfinished rounds and in open positions. A quarter of the rate costs three
  quarters of the wash volume.
- Multi-unit backgrounds: honest orders average 2.5 units. Partial fills make the average filled
  leg 1.5 units.

Full run: 200 seeds, 28,000 runs, 0 errors, 31 minutes with 15 jobs. Detection results are in
`detection/v5/README.md`.

## Quick start

From the repository root:

```bash
python "capstone codes/washtrade/v8/run_smoke.py"                          # ends with RESULT: ALL PASS
python "capstone codes/washtrade/v8/generate.py" list
python "capstone codes/washtrade/v8/generate.py" all --jobs 15 --out <data root>
```

`all` runs every sweep in parallel shard processes (logs in `<dataset>/logs`), then merges each
sweep. The full 200-seed design (28,000 runs) takes about 30 minutes with 15 jobs and needs a few
GB of disk. `run` / `merge` / `hash` / `compare` work as in v7, including Slurm array variables
for `run`. Output goes to `<data root>/ds_v8_<sweep>/`; the default data root is `v8/data`, or set
`WASH_V8_DATA`.
