# Wash-trading mechanism v4: random limit prices

Version 4 keeps v3's wallets, directed cycles, event timing, `gap`, camouflage,
matching, and ground-truth labels. It changes how each event sets its two limit
prices. A wash group now has three additional parameters:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `price_range_fraction` | `0.5` | The permitted radius as a fraction of the current half-spread. `0` reproduces v3 midpoint pricing. |
| `price_sigma_fraction` | `0.5` | Standard deviation of the sell-limit offset as a fraction of that radius. |
| `price_width_fraction` | `0.25` | Largest crossing gap between buy and sell limits as a fraction of the radius. |
| `fallback_range` | `100.0` | Absolute radius reference if the book is empty or one-sided. |

At event start, the controller reads the bid and ask. With both quotes present it
defines `midpoint = (bid + ask) / 2` and
`radius = (ask - bid) / 2 * price_range_fraction`. The sell-limit offset is sampled
from a normal distribution centered at zero and truncated to `[-radius, radius]`.
The buy limit is the sell limit plus a uniform random width bounded by
`price_width_fraction * radius` and the top of the permitted range. Both limits
therefore remain inside the existing spread and cross each other. The sell-limit
distribution is symmetric around the midpoint so the default four-heap clearing
rule does not impose a directional price bias. With `gap > 0`, the first leg rests
until the second is posted; both prices are fixed at event start.

The pricing generator has its own seeded RNG. Changing these price parameters does
not alter event timing or the choice of which leg is posted first. When the book is
empty, the random range is centered on the fundamental; with only one quote, the
range is narrowed to keep the new limit on the non-crossing side of that quote.
`eps` must be zero in v4 because the crossing width now has an explicit parameter.

From the repository root:

```bash
.venv/Scripts/python.exe "capstone codes/washtrade/v4/run_smoke.py"
.venv/Scripts/python.exe "capstone codes/washtrade/v4/generate.py" --name ds_v4 --seeds 10
```

The named scenarios include `triad_midpoint`, `triad_narrow`, `triad`, and
`triad_wide`, which differ only in their price range. Exported data uses schema
`washtrade.v4/1`; `labels/wash_events.csv` records the sampled limits, midpoint,
offset, range, and width. The detector still sees only `observable/`.
