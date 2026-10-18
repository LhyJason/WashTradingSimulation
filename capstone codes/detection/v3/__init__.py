r"""detection.v3 -- the Sethi detector (v1, scored with v2's evaluation) on the v7 sweeps.

v3 adds no new detector. It runs v2's per-run detection + scoring (``detection.v2.experiment
.detect_one``, configs paper / no_propagation / theta_hi_1) over the five v7 sweeps, where the
wash adversary rotates wallets from a pool, splits rounds into random-size unit trades and leaves
residual positions, and adds control-calibrated thresholds (``calibrate.py``).
"""
