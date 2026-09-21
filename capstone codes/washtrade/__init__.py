"""washtrade -- wash-trading agents / controllers built on PyMarketSim (capstone Part 1).

This is a container package; concrete mechanism versions live in subpackages (v1, v2, ...)
so designs can evolve without breaking earlier, already-run experiments.

Not importable as ``capstone codes.washtrade`` because the container folder name contains a
space; entry points add ``<repo>/capstone codes`` to sys.path and then ``import washtrade.v1``.
"""
