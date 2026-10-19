"""Seed-level bootstrap CIs for the final detector (v4 vs v6), from the existing per-run results.

Unit of resampling = seed (all cells of one seed move together is NOT assumed; seeds are drawn
independently within each group). Pooled ratio = sum(tp)/sum(pos) over runs, owner label.
"""
import os, sys
import numpy as np, pandas as pd
DATA = sys.argv[2] if len(sys.argv) > 2 else os.environ["WASH_V8_DATA"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
B = 2000
rng = np.random.default_rng(0)
cols = ["cell", "seed", "background", "f_scenario", "f_gap", "f_edge_order", "f_pipeline",
        "f_lam_wash", "method", "trade_pos_owner", "trade_tp_owner", "trade_neg_owner",
        "trade_fp_owner"]

def boot(g, num, den):
    """pooled ratio num/den with percentile CI, resampling seeds."""
    s = g.groupby("seed")[[num, den]].sum()
    n, d = s[num].to_numpy(float), s[den].to_numpy(float)
    if d.sum() == 0:
        return np.nan, np.nan, np.nan
    pt = n.sum() / d.sum()
    idx = rng.integers(0, len(s), (B, len(s)))
    dd = d[idx].sum(1); nn = n[idx].sum(1)
    ok = dd > 0
    r = nn[ok] / dd[ok]
    return pt, np.percentile(r, 2.5), np.percentile(r, 97.5)

def table(df, keys, methods, kind):
    rows = []
    for k, g in df[df.method.isin(methods)].groupby(keys + ["method"]):
        num, den = ("trade_tp_owner", "trade_pos_owner") if kind == "recall" else ("trade_fp_owner", "trade_neg_owner")
        p, lo, hi = boot(g, num, den)
        rows.append(dict(zip(keys + ["method"], k)) | dict(metric=kind, est=p, lo=lo, hi=hi, n_seeds=g.seed.nunique()))
    return pd.DataFrame(rows)

out = []
for sweep in ["adversary", "honest"]:
    d = pd.read_csv(fr"{DATA}\ds_v8_{sweep}\detection_v6\results.csv", usecols=lambda c: c in cols)
    d["sweep"] = sweep
    w, c = d[d.f_scenario != "control"], d[d.f_scenario == "control"]
    if sweep == "adversary":
        w = w.copy(); w["gap"] = w.f_gap
        groupings = [["sweep", "background"], ["sweep", "background", "gap"],
                     ["sweep", "background", "f_pipeline"], ["sweep", "background", "f_edge_order"],
                     ["sweep", "background", "f_lam_wash"]]
    else:
        w = w.copy(); w["gap"] = np.nan
        groupings = [["sweep", "cell"]]
    for keys in groupings:
        out.append(table(w, keys, ["v4", "v6"], "recall").assign(group="+".join(keys[1:])))
    ck = ["sweep", "cell"] if sweep == "honest" else ["sweep", "background"]
    out.append(table(c, ck, ["v4", "v6"], "fpr").assign(group="control_" + "+".join(ck[1:])))
    out.append(table(w, ck, ["v4", "v6"], "fpr").assign(group="washrun_" + "+".join(ck[1:])))
res = pd.concat(out, ignore_index=True)
res.to_csv(f"{OUT}/seed_ci{sys.argv[3] if len(sys.argv) > 3 else ''}.csv", index=False)
print(len(res))
