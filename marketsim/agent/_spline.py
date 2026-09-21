"""Drop-in replacement for ``fastcubicspline.FCS``.

The upstream code depends on the ``fastcubicspline`` package, which ships a
Cython/C extension that fails to build on Python >= 3.11 (and needs a C
compiler on Windows). The HBL agent only ever constructs an ``FCS`` over an
interval ``[x_low, x_high]`` from a sequence of ``y`` values sampled at
equidistant nodes, and then evaluates it at scalar prices inside that interval.

This shim reproduces that behaviour exactly using NumPy/SciPy:
  * with 2 sample points a cubic spline through them is just the straight line,
    so linear interpolation is mathematically identical;
  * with >= 4 points we fall back to SciPy's ``CubicSpline`` to stay faithful
    to the "cubic spline interpolation" described in the paper.
"""

import numpy as np

try:
    from scipy.interpolate import CubicSpline
except Exception:  # pragma: no cover - scipy is a hard dependency anyway
    CubicSpline = None


class FCS:
    def __init__(self, x_low, x_high, y, *args, **kwargs):
        self.x_low = float(x_low)
        self.x_high = float(x_high)
        self.y = np.asarray(y, dtype=float)
        n = len(self.y)
        if self.x_high == self.x_low or n < 2:
            self._x = np.array([self.x_low])
            self._spline = None
        else:
            self._x = np.linspace(self.x_low, self.x_high, n)
            if n >= 4 and CubicSpline is not None:
                self._spline = CubicSpline(self._x, self.y)
            else:
                self._spline = None  # use linear interpolation

    def __call__(self, x):
        if self._spline is not None:
            return float(self._spline(x))
        if len(self.y) == 1:
            return float(self.y[0])
        return float(np.interp(x, self._x, self.y))
