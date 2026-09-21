r"""
run_2hbl_13zi.py -- run a "2 HBL + 13 ZI" market with ComposableSimulator

Run (from the repo root):
  python -m repro.composable.run_2hbl_13zi --num-sims 2000

Or change the mix / parameters:
  python -m repro.composable.run_2hbl_13zi --zi 13 --hbl 2 --lam 5e-3

Notes:
  * This is a "custom composition" experiment, N=15 (!= the paper's Table 3 N=25), so the
    per-agent profit will not equal the Table 3 numbers; the point is to compare HBL vs ZI
    profit within the same market (paper §5.3 reports TRON's edge over ZI; similar idea).
  * Fixed environment parameters default to paper Table 3: q_max=10, f_bar=1e5, kappa=0.01,
    T=2000; the variable ones lambda / sigma_s^2 / sigma_pv^2 default to a Table 3 (env B)
    setting and can be overridden on the command line.
"""
import sys
import argparse
from collections import defaultdict

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tqdm import tqdm

from repro.composable.simulator import ComposableSimulator


def run_one(args):
    """Run one simulation; return {type name: [final wealth of each agent of that type]}."""
    sim = ComposableSimulator(
        agent_composition={"ZI": args.zi, "HBL": args.hbl},
        sim_time=args.sim_time,
        lam=args.lam,
        mean=args.mean,
        r=args.r,
        shock_var=args.shock_var,
        q_max=args.q_max,
        pv_var=args.pv_var,
        shade=args.shade,
        eta=args.eta,
        L=args.L,
    )
    sim.run()
    values = sim.final_values()                # {agent_id: wealth}
    by_type = defaultdict(list)
    for aid, v in values.items():
        by_type[sim.agent_types[aid]].append(v)
    return by_type


def main():
    p = argparse.ArgumentParser(description="2 HBL + 13 ZI composition simulation")
    p.add_argument("--zi", type=int, default=13, help="number of ZI agents")
    p.add_argument("--hbl", type=int, default=2, help="number of HBL agents")
    p.add_argument("--num-sims", type=int, default=2000, help="number of simulations")
    p.add_argument("--sim-time", type=int, default=2000, help="T total time steps")
    p.add_argument("--lam", type=float, default=5e-3, help="lambda reentry rate")
    p.add_argument("--mean", type=float, default=1e5, help="f_bar fundamental mean")
    p.add_argument("--r", type=float, default=0.01, help="kappa mean-reversion speed (called r in code)")
    p.add_argument("--shock-var", type=float, default=1e6, help="sigma_s^2 shock variance")
    p.add_argument("--q-max", type=int, default=10, help="q_max maximum holding")
    p.add_argument("--pv-var", type=float, default=5e6, help="sigma_pv^2 private-value variance")
    p.add_argument("--shade", type=float, nargs=2, default=[250, 500],
                   metavar=("RMIN", "RMAX"), help="[R_min, R_max] surplus range")
    p.add_argument("--eta", type=float, default=0.2, help="ZI threshold eta")
    p.add_argument("--L", type=int, default=4, help="HBL memory length L")
    p.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    args = p.parse_args()

    if args.seed is not None:
        import random
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"Composition: {args.zi} ZI + {args.hbl} HBL  (N={args.zi + args.hbl})")
    print(f"T={args.sim_time}  lam={args.lam}  r(kappa)={args.r}  "
          f"shock_var={args.shock_var:g}  pv_var={args.pv_var:g}")
    print(f"shade={args.shade}  eta={args.eta}  L={args.L}  num_sims={args.num_sims}")

    # Accumulate, per simulation, the per-type mean profit
    per_type_means = defaultdict(list)   # type -> [per-sim mean of that type]
    all_means = []                       # per-sim mean over all agents
    for _ in tqdm(range(args.num_sims), desc="simulating"):
        by_type = run_one(args)
        run_all = []
        for t, vals in by_type.items():
            per_type_means[t].append(float(np.mean(vals)))
            run_all.extend(vals)
        all_means.append(float(np.mean(run_all)))

    print(f"\nResults (across {args.num_sims} simulations, one vote per agent):")
    for t in sorted(per_type_means):
        arr = np.array(per_type_means[t])
        mean = arr.mean()
        sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else float("nan")
        print(f"  {t:>4} mean profit = {mean:10.2f}  +/-  {sem:7.2f} (s.e.)")

    allm = np.array(all_means)
    print(f"  {'ALL':>4} mean profit = {allm.mean():10.2f}  +/-  "
          f"{allm.std(ddof=1) / np.sqrt(len(allm)):7.2f} (s.e.)")


if __name__ == "__main__":
    main()
