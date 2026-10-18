r"""detection.v4 -- a multi-channel version of the Sethi detector, on the v7 sweeps.

v3 showed that Sethi et al.'s closing score fails in two ways. In active markets honest wallets
close as often as random walks do. Against a rotating wallet pool, wash wallets close only about
once. v4 keeps the network step (Part II) and adds Part I evidence channels that look at short
time windows:

  * ``channels.py``    closure (v1, unchanged), fast round trips, same-pair repeats;
  * ``detector.py``    every channel propagated over the counterparty graph -> pair scores;
  * ``fusion.py``      control-run tail probabilities, max-evidence fusion, calibrated thresholds;
  * ``evaluate.py``    per-run, per-method scoring against the labels;
  * ``experiment.py``  pass 1 (control null), pass 2 (every v7 run, sharded) + merge.

See README.md for definitions, outputs and the main results.
"""
