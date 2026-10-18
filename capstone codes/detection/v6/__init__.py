r"""detection.v6 -- detection v4's channels plus a cluster channel, on the washtrade.v8 sweeps.

v5 showed that a pipelined, interleaved, slow adversary removes v4's per-wallet, short-window
evidence cheaply, while the group keeps trading with itself about twice as often as
volume-proportional mixing predicts. v6 finds candidate groups without labels and scores them:

  * ``cluster.py``     Louvain communities of the counterparty graph, wallet excess, pair score;
  * ``detector.py``    ``detect_v6`` = v4's detector + the cluster channel(s);
  * ``fusion.py``      v6's channel and method lists on v4's fusion (control null, floor 0.5);
  * ``experiment.py``  pass 1 (control null), pass 2 (every v8 run, sharded) + merge.

See README.md for definitions, outputs and the main results.
"""
