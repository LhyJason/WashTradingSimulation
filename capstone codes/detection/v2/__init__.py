r"""detection.v2 -- the Sirolly-Ma-Kanoria-Sethi detector on the washtrade.v6 sweeps.

The detector itself is v1's (pairing, closures, propagation, Algorithms 1 and 2), imported
unchanged. v2 adds what the v6 study needs:

  * ``evaluate.py``    v1's metrics + same-group truth (cross-group fills), a per-group table
                       comparing each wash group's mean score with the leakage formula
                       x_bar = (x0_bar + l * s_bar) / (1 + l), and threshold curves
                       (counts at theta = 0.50 ... 1.00) for later threshold rules;
  * ``closing.py``     the random-walk benchmark for honest closures and a per-wallet table;
  * ``experiment.py``  sharded, resumable runs over every v6 sweep + merge.

See README.md for outputs, definitions and the main results.
"""
