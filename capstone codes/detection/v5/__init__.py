r"""detection.v5 -- detection v4's detector on the washtrade.v8 sweeps.

v5 adds no new detector. It runs v4 unchanged (channels, propagation, control-null fusion,
floor 0.5, exact thresholds) on the v8 data, where the adversary hides from short time windows
(interleaved edges, a pipeline of open rounds, a slower wash rate) and honest traders post
multi-unit orders. Control thresholds are recalibrated per v8 background. See README.md.
"""
