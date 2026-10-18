# Detection v6: a cluster channel for wash groups

Detection v5 showed that v8's adaptive adversary removes v4's evidence cheaply in the active
markets. It overlaps its rounds (a pipeline of D open rounds), interleaves edges and slows down,
so no wallet undoes its position or repeats a pair within v4's short windows. With D = 8 and
interleaved edges, v4's recall in active fell to 0.07.

What the adversary cannot change is the defining property of wash trading: no beneficial
ownership changes, so the group's volume stays inside the group. v5 measured it with the labels:
in active, the group trades 60% of its volume internally, against 31% if its counterparties were
drawn in proportion to volume, and the execution knobs leave that unchanged. v6 adds a channel
that looks for this without labels.

## The cluster channel (`cluster.py`)

1. **Communities.** Louvain (networkx, modularity, fixed seed) on the counterparty graph V, the
   pro-rata volumes of v1's pairing. Resolution 1 (networkx's default) is the main setting; 2 is a
   sensitivity check.
2. **Wallet score.** For wallet $i$ in community $C$, with $a_i$ the share of its volume inside
   $C$ and $e_i = (\mathrm{vol}(C) - \mathrm{vol}_i) / (\mathrm{tot} - \mathrm{vol}_i)$ the same
   share under volume-proportional mixing:

   $$s_i = \frac{a_i - e_i}{\max(1 - e_i,\ a_i)} = \min\Big(\frac{a_i - e_i}{1 - e_i},\ 1 - \frac{e_i}{a_i}\Big).$$

   The first form is the share of the possible excess that is realised. The second is the share
   of the in-community volume beyond mixing ($a/e$ is the lift). At the floor 0.5 a wallet needs
   both: half of the possible excess and twice the mixing share.
3. **Pair score.** $\min(s_i, s_j)$, clipped at 0, when both wallets are in the same community; 0
   otherwise; NaN for self-pairs. The cluster score is not propagated: the community already is
   the network context.
4. **Fusion** (`fusion.py`). The cluster channel joins v4's seven channels under v4's rules,
   unchanged: control tail probability, floor 0.5, max-evidence fusion, exact thresholds at a
   pooled control FPR of 1%. `v6` = v4's channels + `cluster`; `v6_r2` uses resolution 2.
5. **Where the channel applies.** It tests "this group trades with itself more than a mixing
   market predicts", which needs a mixing honest market in which the group is a minority. When a
   group carries most of the volume (thin and thick: 56–88%), the honest wallets form the small,
   closed block, and no graph statistic can tell which block is the wash group. The channel is
   used only where the control runs mix: mean modularity of the control partitions below 0.3.
   Elsewhere every cluster score is set to 0, and `v6` equals `v4` exactly. The rule uses control
   runs only, no labels.

| Background | Control modularity | Cluster channel |
|---|---|---|
| active / active_mu4 | 0.149 / 0.163 | on |
| active_dense / active_dense_mu4 | 0.174 / 0.189 | on |
| thick / thick_mu4 | 0.389 / 0.470 | off |
| thin / thin_mu4 | 0.420 / 0.469 | off |

## How the score was chosen

Screened on seeds 0–9 of the `adversary` wash cells and seeds 0–19 of the `honest` wash cells,
with 100 control runs per background. Each step fixed a failure of the previous one:

- **Louvain finds the group.** 92–98% of same-owner volume has both wallets in one community, for
  every execution. Resolution 0.5 puts each active control run into a single community and was
  dropped.
- **Scoring whole communities** by their internal share flagged about 40% of honest volume in wash
  runs at a 1% control FPR. About 60 pool wallets place only a camouflage order or two. Each sits
  in its honest counterparty's community and makes it look closed. Wallet scores with the pair
  minimum fixed this.
- **Removing the most anomalous community and rescoring the rest** (peeling) did not help.
- **(a − e)/(1 − e) alone** failed when rounds are block-executed against single-unit honest
  orders (`honest` sweep, `evasive_block`, active). The group then carries 48% of all volume.
  Louvain puts the honest market into one community with 53–57% of the volume and an internal
  share of 0.86–0.88, which scores about 0.71, the same as the wash communities. About half of
  the honest volume was flagged. The lift caps a community that holds half the market at 0.5.
- **The lift alone** is noisy for small communities (FPR in active wash runs 0.9% → 5.9%), hence
  the minimum.
- **The minimum switched the channel on in thin and thick.** With the first form alone, their
  control thresholds were 1, so it never fired. With the minimum it fired and raised the FPR in
  thin wash runs from 1.4% to 10%, for almost no recall. This led to the applicability rule (5.).

The floor (0.5) and the main resolution (1) are v4's and networkx's defaults, not tuned. The full
run uses all 200 seeds. Seeds 0–9 alone and seeds 10–199 alone agree within 0.03 on every pooled
recall, precision and FPR of v6 (seeds 0–9 are only 10 seeds per cell).

## Files

| File | Content |
|---|---|
| `cluster.py` | Louvain communities, wallet score, pair score, modularity |
| `detector.py` | `detect_v6`: `detect_v4` plus the cluster channel at each resolution |
| `fusion.py` | v6's channel and method lists on v4's `Null`, the applicability rule |
| `experiment.py` | Pass 1 `null` (honest sweep's controls), pass 2 `run`/`merge` over the v8 sweeps, `all` |
| `run_smoke.py` | Cluster channel on hand-made graphs, v4 unchanged, null and applicability, runner, owner label; ends with `RESULT: ALL PASS` |
| `results_detection_v6.ipynb` | Results: applicability and thresholds, recall by execution, precision / FPR / AP, the `honest` sweep, sensitivity, held-out seeds. It only reads results; set `WASH_V8_DATA` first. `RUN_DETECTION = True` reruns everything |

v4's `Null`, `thresholds` and `score_run` now take the channel and method lists as parameters.
They default to v4's, so detection v4 and v5 are unchanged (their smoke tests pass, and v6's
thresholds for v4's methods equal v5's).

## Running

From the repository root. The full run takes about 16 minutes with 12 jobs (15 jobs ran out of
memory on 16 GB during the detection v3 rescore).

```bash
python "capstone codes/detection/v6/run_smoke.py"
python "capstone codes/detection/v6/experiment.py" all --jobs 12 --data <v8 data root>
```

Outputs:

| File | Content |
|---|---|
| `<data>/detection_v6_null.csv` | one row per control pair: qty, every single score, the run's modularity |
| `<data>/detection_v6_thresholds.csv` | (background, method, floor): θ and pooled control FPR per target |
| `<data>/detection_v6_backgrounds.csv` | control modularity and whether the cluster channel is used |
| `ds_v8_<sweep>/detection_v6/results.csv` | (run, method): v4's columns (owner label included), `ambiguous_volume_share`, `n_comm_*` and `modularity_*` per resolution |
| `ds_v8_honest/detection_v6/wallets.csv` | (run, wallet): v4's columns plus `comm_cluster`, `x_cluster` (wallet score) per resolution |

`merge` now reads the part files with round-trip float parsing, so the merged file equals the parts
exactly. Earlier versions' merges change the last digit of some floats.

## Main results (200 seeds per cell; pooled control FPR 1%, floor 0.5)

Owner label, with the order label in parentheses. The active markets are the main baseline. In
thin and thick the cluster channel is off and v6 is v4.

**Recall by execution, v4 → v6** (`adversary` sweep, active):

| Execution | lam_wash | gap 0 | gap 5 |
|---|---|---|---|
| sequential, D = 1 (v7) | 0.05 | 0.59 → 0.85 | 0.06 → 0.48 |
| interleaved, D = 1 | 0.05 | 0.76 → 0.90 | 0.30 → 0.59 |
| sequential, D = 4 | 0.05 | 0.42 → 0.83 | 0.03 → 0.46 |
| interleaved, D = 4 | 0.05 | 0.15 → 0.83 | 0.02 → 0.46 |
| sequential, D = 8 | 0.05 | 0.18 → 0.81 | 0.00 → 0.46 |
| interleaved, D = 8 | 0.05 | 0.07 → 0.81 | 0.01 → 0.45 |
| sequential, D = 1 | 0.0125 | 0.32 → 0.82 | 0.02 → 0.53 |
| sequential, D = 8 | 0.0125 | 0.01 → 0.74 | 0.00 → 0.59 |
| interleaved, D = 8 | 0.0125 | 0.03 → 0.69 | 0.01 → 0.56 |

In active_dense the same rows give 0.62–0.85 at gap 0 (v4: 0.02–0.69) and 0.27–0.50 at gap 5
(v4: 0.01–0.20).

**Pooled over every execution** (`adversary` sweep):

| | active | active_dense |
|---|---|---|
| recall gap 0, v4 → v6 | 0.30 → 0.82 (0.34 → 0.86) | 0.25 → 0.73 (0.28 → 0.75) |
| recall gap 5, v4 → v6 | 0.05 → 0.50 (0.08 → 0.51) | 0.05 → 0.34 (0.07 → 0.30) |
| precision gap 0, v4 → v6 | 0.98 → 0.93 (0.97 → 0.84) | 0.92 → 0.93 (0.91 → 0.85) |
| precision gap 5, v4 → v6 | 0.92 → 0.86 (0.89 → 0.59) | 0.57 → 0.72 (0.51 → 0.40) |
| FPR in wash runs gap 0, v4 → v6 | 0.1% → 0.9% (0.1% → 2.1%) | 0.1% → 0.4% (0.2% → 0.8%) |
| v6 FPR gap 0: touch / far | 2.5% / 0.3% | 1.4% / 0.1% |
| AP gap 0, v4 → v6 | 0.57 → 0.92 | 0.52 → 0.89 |
| wallet recall / FPR gap 0, v6 | 0.41 / 4.6% | 0.32 / 2.8% |

1. **The adaptive adversary no longer evades in active markets.** Every execution is caught at
   0.69–0.90 at gap 0, against 0.01–0.76 for v4. The pipeline, interleaving and slow rate change
   when the group trades, not with whom.
2. **Gap 5 is no longer hopeless.** A gap delays the second leg of an event, so honest traders
   fill some of the legs (interception). The remaining legs still connect group wallets. Recall in
   active rises from at most 0.30 to 0.45–0.59.
3. **The cost is small and sits next to the group.** Precision stays at 0.93 at gap 0. The extra
   false positives are honest volume that touches a wash wallet (2.5%). Far from the group the FPR
   is 0.3%, below the control 1%. Under the order label precision looks lower (0.84): the cluster
   channel catches camouflage volume between group wallets, which the order label calls honest.
4. **Ranking improves the most.** AP rises from 0.57 to 0.92 in active.
5. **Multi-unit honest orders** (`honest` sweep). v6's recall in active_mu4 / active_dense_mu4 is
   0.71 / 0.58 against v7's execution (v4: 0.33 / 0.19), and 0.65 / 0.54 against interleaved
   D = 8 (v4: 0.00 / 0.00).
6. **Block-executed rounds** (`honest` sweep, `evasive_block`): recall 0.95 (active) at an FPR in
   wash runs of 3.0%, 1.8% far from the group. This is the case that broke the first form of the
   score.
7. **Resolution 2** gives nearly the same recall (active gap 0: 0.79) at higher precision (0.99)
   and an FPR of 0.1%. **Floor** (order label: owner counts are kept at the main floor only): 0.3
   gives 0.94 at 4.8% in active, 0.7 gives 0.52 at 0.9%.

## Limitations

- **Thin and thick markets are not helped.** There the wash group carries most of the volume, so
  community structure cannot tell the group from the market. v4 covers these markets (recall
  0.65–0.68 at gap 0), but the pipeline with a slow rate still evades it there (0.05–0.06).
- **The adversary does not know about this channel.** A cluster-aware adversary could lower its
  internal share only by trading more with honest traders, which is real trading with real
  inventory risk, or by becoming a larger share of the market, which the lift caps. Testing such
  an adversary is the natural next step.
- **Wallet recall is low** (0.41 in active). Most of the 96 pool wallets only place camouflage
  orders, which look like honest trading; the wallet label still counts them as group members.
- **Louvain is randomized.** Results use one fixed seed; the sensitivity to the seed is not
  measured.
