"""Draw two random crossing limit prices per event, preserving v3 scheduling."""

import math

import numpy as np

from washtrade.v3.controller import WashController as V3WashController


class WashController(V3WashController):
    _PRICE_SEED_OFFSET = 15427

    def __init__(self, *args, price_range_fraction=0.5, price_sigma_fraction=0.5,
                 price_width_fraction=0.25, fallback_range=100.0, **kwargs):
        if not 0 <= price_range_fraction < 1:
            raise ValueError("price_range_fraction must be in [0, 1)")
        if price_sigma_fraction <= 0:
            raise ValueError("price_sigma_fraction must be positive")
        if not 0 <= price_width_fraction <= 1:
            raise ValueError("price_width_fraction must be in [0, 1]")
        if fallback_range <= 0:
            raise ValueError("fallback_range must be positive")
        if kwargs.get("eps", 0.0) != 0:
            raise ValueError("v4 uses one random price for both legs; eps must be 0")
        seed = kwargs.get("seed")
        super().__init__(*args, **kwargs)
        self.price_range_fraction = float(price_range_fraction)
        self.price_sigma_fraction = float(price_sigma_fraction)
        self.price_width_fraction = float(price_width_fraction)
        self.fallback_range = float(fallback_range)
        # Price draws do not change the event times or the first-leg coin flips.
        self._price_rng = np.random.default_rng(
            None if seed is None else int(seed) + self._PRICE_SEED_OFFSET)

    def _anchor_and_delta(self):
        midpoint, _, bb, ba = super()._anchor_and_delta()
        if math.isfinite(bb) and math.isfinite(ba):
            radius = (ba - bb) * 0.5 * self.price_range_fraction
        elif math.isfinite(bb):
            radius = min(self.fallback_range, midpoint - bb) * self.price_range_fraction
        elif math.isfinite(ba):
            radius = min(self.fallback_range, ba - midpoint) * self.price_range_fraction
        else:
            radius = self.fallback_range * self.price_range_fraction
        radius = max(0.0, radius)
        if radius == 0:
            sell_offset = 0.0
        else:
            sigma = radius * self.price_sigma_fraction
            # Rejection sampling avoids artificial masses at the range boundaries.
            sell_offset = self._price_rng.normal(0.0, sigma)
            while abs(sell_offset) > radius:
                sell_offset = self._price_rng.normal(0.0, sigma)
        # The sell limit determines this market's clear price. Draw it symmetrically
        # around the midpoint, then give the buy limit positive crossing room.
        max_width = min(radius * self.price_width_fraction, radius - sell_offset)
        width = (self._price_rng.uniform(0.0, max_width)
                 if max_width > 0 else 0.0)
        sell_price = midpoint + sell_offset
        buy_price = sell_price + width
        self._midpoint_anchor = float(midpoint)
        self._price_radius = float(radius)
        self._price_width = float(width)
        return float((sell_price + buy_price) / 2), float(width / 2), bb, ba

    def _post_open(self, t):
        posts = super()._post_open(t)
        self._open["midpoint_anchor"] = self._midpoint_anchor
        self._open["price_radius"] = self._price_radius
        self._open["price_offset"] = self._open["sell_px"] - self._midpoint_anchor
        self._open["price_width"] = self._price_width
        return posts
