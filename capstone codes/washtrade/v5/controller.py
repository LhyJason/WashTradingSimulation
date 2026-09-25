"""Microprice-centered event pricing with spread, volatility, and tick constraints."""

import math

import numpy as np

from washtrade.v3.controller import WashController as V3WashController


class WashController(V3WashController):
    _PRICE_SEED_OFFSET = 22109

    def __init__(self, *args, price_range_fraction=0.8, price_sigma_fraction=0.5,
                 volatility_multiplier=0.5, volatility_lookback=20,
                 max_abs_range=250.0, fallback_range=100.0, tick_size=1.0,
                 use_microprice=True, **kwargs):
        if not 0 <= price_range_fraction < 1:
            raise ValueError("price_range_fraction must be in [0, 1)")
        if price_sigma_fraction <= 0:
            raise ValueError("price_sigma_fraction must be positive")
        if volatility_multiplier <= 0 or volatility_lookback < 3:
            raise ValueError("invalid volatility parameters")
        if max_abs_range <= 0 or fallback_range <= 0 or tick_size <= 0:
            raise ValueError("range and tick parameters must be positive")
        if kwargs.get("eps", 0.0) != 0:
            raise ValueError("v5 samples one common event price; eps must be 0")
        seed = kwargs.get("seed")
        super().__init__(*args, **kwargs)
        self.price_range_fraction = float(price_range_fraction)
        self.price_sigma_fraction = float(price_sigma_fraction)
        self.volatility_multiplier = float(volatility_multiplier)
        self.volatility_lookback = int(volatility_lookback)
        self.max_abs_range = float(max_abs_range)
        self.fallback_range = float(fallback_range)
        self.tick_size = float(tick_size)
        self.use_microprice = bool(use_microprice)
        self._price_rng = np.random.default_rng(
            None if seed is None else int(seed) + self._PRICE_SEED_OFFSET)

    @staticmethod
    def _best_depth(queue, best):
        if not math.isfinite(best):
            return 0.0
        return float(sum(order.quantity for order in queue.order_dict.values()
                         if order.quantity > 0 and float(order.price) == float(best)))

    def _recent_volatility(self):
        # matched_orders contains two legs per transaction. Collapse to one price per clear.
        by_time = {}
        for matched in self.market.matched_orders:
            by_time[matched.time] = float(matched.price)
        prices = list(by_time.values())[-(self.volatility_lookback + 1):]
        if len(prices) < 4:
            return math.nan
        return float(np.std(np.diff(prices), ddof=1))

    def _round_inside(self, value, low, high):
        tick = self.tick_size
        low_tick = math.ceil(low / tick) * tick
        high_tick = math.floor(high / tick) * tick
        if low_tick > high_tick:
            return min(max(round(value / tick) * tick, low), high)
        return min(max(round(value / tick) * tick, low_tick), high_tick)

    def _anchor_and_delta(self):
        fallback_center, _, bb, ba = super()._anchor_and_delta()
        ob = self.market.order_book
        bid_depth = self._best_depth(ob.buy_unmatched, bb)
        ask_depth = self._best_depth(ob.sell_unmatched, ba)
        two_sided = math.isfinite(bb) and math.isfinite(ba)

        if two_sided:
            midpoint = 0.5 * (bb + ba)
            if self.use_microprice and bid_depth + ask_depth > 0:
                center = (ba * bid_depth + bb * ask_depth) / (bid_depth + ask_depth)
            else:
                center = midpoint
            spread_radius = 0.5 * (ba - bb) * self.price_range_fraction
            low_book = bb + self.tick_size
            high_book = ba - self.tick_size
        else:
            midpoint = fallback_center
            center = fallback_center
            spread_radius = self.fallback_range * self.price_range_fraction
            low_book = bb + self.tick_size if math.isfinite(bb) else -math.inf
            high_book = ba - self.tick_size if math.isfinite(ba) else math.inf

        volatility = self._recent_volatility()
        volatility_cap = (self.volatility_multiplier * volatility
                          if math.isfinite(volatility) and volatility > 0 else math.inf)
        radius = min(spread_radius, volatility_cap, self.max_abs_range)
        radius = max(0.0, radius)
        low = max(center - radius, low_book)
        high = min(center + radius, high_book)
        if low > high:
            low = high = min(max(center, low_book), high_book)

        if high == low:
            sampled = low
        else:
            sigma = max(self.tick_size, radius * self.price_sigma_fraction)
            sampled = self._price_rng.normal(center, sigma)
            while sampled < low or sampled > high:
                sampled = self._price_rng.normal(center, sigma)
        target = self._round_inside(sampled, low, high)

        self._pricing_meta = dict(
            midpoint_anchor=float(midpoint), microprice_anchor=float(center),
            bid_depth_at_open=bid_depth, ask_depth_at_open=ask_depth,
            book_imbalance=((bid_depth - ask_depth) / (bid_depth + ask_depth)
                            if bid_depth + ask_depth else math.nan),
            recent_volatility=volatility, price_radius=float(radius),
            price_low=float(low), price_high=float(high), target_price=float(target),
            midpoint_offset=float(target - midpoint),
            microprice_offset=float(target - center), tick_size=self.tick_size,
        )
        # A common random target avoids a deterministic side-dependent price impact.
        return float(target), 0.0, bb, ba

    def _post_open(self, t):
        posts = super()._post_open(t)
        self._open.update(self._pricing_meta)
        return posts
