# Wash-trading mechanism v5: microprice-centered random prices

Version 5 keeps v3's directed wallet cycles, event timing, gap, camouflage,
matching, and labels. Each wash event now draws one common buy/sell limit price
from a truncated normal distribution and rounds it to the market tick.

When both quotes exist, the center is the top-of-book microprice:

```text
(best_ask * bid_depth + best_bid * ask_depth) / (bid_depth + ask_depth)
```

The nominal radius is `half_spread * price_range_fraction`. It is capped by
`volatility_multiplier * std(recent clear-to-clear price changes)` and by
`max_abs_range`. The actual bounds are clipped inside the current best bid and
ask. If the book is empty or one-sided, the v3 fundamental/quote fallback is used.

The selected defaults after a 27-configuration sweep are:

| Parameter | Default |
| --- | ---: |
| `price_range_fraction` | `0.8` |
| `price_sigma_fraction` | `0.5` |
| `volatility_multiplier` | `0.5` |
| `volatility_lookback` | `20` |
| `max_abs_range` | `250.0` |
| `fallback_range` | `100.0` |
| `tick_size` | `1.0` |

These settings retained a pooled 87.39% self-trade rate across thin/thick and
gap 0/5/20 experiments, versus 87.39% for v3 midpoint pricing. Only 1.79% of
two-sided events were within half a tick of the microprice, compared with every
event being exactly at the midpoint in v3.

The sweep crossed three values for each of range fraction (`0.2, 0.5, 0.8`),
sigma fraction (`0.25, 0.5, 0.8`), and volatility multiplier (`0.5, 1.0,
2.0`). Each configuration used 15 seeds, both liquidity backgrounds, and the
gap 0/5/20 scenarios: 90 runs per configuration and 2,430 runs in total. The
selected configuration was within 0.08 percentage points of the highest
self-trade rate, had the lowest intercept rate among the leading
configurations, and produced materially noncentral prices. The complete
aggregate results are in `parameter_sweep_summary.csv`.
