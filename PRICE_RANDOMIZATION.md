# Wash Trade Price Randomization

## Motivation

In v3, both wash orders are placed at the midpoint:

```text
midpoint = (best bid + best ask) / 2
```

This makes the simulated orders easy to distinguish because every wash trade
uses the same price rule. The new version adds small, market-dependent price
variation while keeping the existing wallet cycles and matching logic.

## Method

When both sides of the order book are available, the price is centered on the
top-of-book microprice:

```text
microprice = (best ask * bid depth + best bid * ask depth)
             / (bid depth + ask depth)
```

The allowed radius is:

```text
radius = min(0.8 * half spread, 0.5 * recent volatility, 250)
```

The radius is also clipped inside the best bid and ask. A price is drawn from a
truncated normal distribution centered on the microprice, with standard
deviation `0.5 * radius`, and rounded to a tick size of 1. The buy and sell legs
of one wash event use the same sampled price so that they can still match.

## Parameter Selection

We tested 27 parameter combinations using:

- 15 random seeds;
- thin and thick liquidity backgrounds;
- order gaps of 0, 5, and 20 steps.

This produced 2,430 simulation runs. The selected defaults are:

```text
price_range_fraction  = 0.8
price_sigma_fraction  = 0.5
volatility_multiplier = 0.5
volatility_lookback   = 20
max_abs_range         = 250
tick_size             = 1
```

## Results

| Metric | v3 midpoint | Randomized price |
| --- | ---: | ---: |
| Self-trade rate | 87.39% | 87.39% |
| Interception rate | 9.63% | 9.46% |
| Wash volume share | 65.41% | 65.40% |
| Events near the price center | 100% | 1.79% |

The randomized mechanism preserves the original self-trade rate while removing
the deterministic midpoint pattern. The existing adaptive detector also remains
effective: its F1 score changed from 99.65% to 99.64%.

The full sweep is available in
`capstone codes/washtrade/v5/parameter_sweep_summary.csv`.
