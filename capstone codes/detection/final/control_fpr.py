"""Run-level false-positive rate of the control runs (no wash group), design seeds vs fresh seeds.

The thresholds put the POOLED control FPR at 1% on the design seeds. This shows how that 1% is
spread over runs, and which channel the false positives come from.

    python control_fpr.py <out dir> [<design data root> <fresh data root>]

The roots default to $WASH_V8_DATA and $WASH_V8_HELDOUT.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
SRC = {"design seeds 0-199": sys.argv[2] if len(sys.argv) > 2 else os.environ["WASH_V8_DATA"],
       "fresh seeds 1000-1099": sys.argv[3] if len(sys.argv) > 3 else os.environ["WASH_V8_HELDOUT"]}
BGS = ["active", "active_dense", "thin", "thick"]
COLS = ["run_id", "seed", "background", "f_scenario", "method", "trade_fp_owner", "trade_neg_owner"]
CHANNELS = ["closure", "roundtrip_100", "repeat_50", "cluster"]

frames = []
for src, root in SRC.items():
    for sw in ("adversary", "honest"):
        d = pd.read_csv(f"{root}/ds_v8_{sw}/detection_v6/results.csv", usecols=COLS)
        d = d[(d.f_scenario == "control") & d.background.isin(BGS)]
        frames.append(d.assign(src=src, sweep=sw))
d = pd.concat(frames)
# the adversary sweep's controls are the honest sweep's standard-background controls bit for bit;
# keep one copy per run
d = d[d.sweep == "honest"] if (d.sweep == "honest").any() else d
d["fpr"] = d.trade_fp_owner / d.trade_neg_owner

rows = []
for (src, bg), g in d.groupby(["src", "background"]):
    v6 = g[g.method == "v6"].set_index("run_id")
    fp, neg = v6.trade_fp_owner, v6.trade_neg_owner
    top5 = fp.sort_values().index[-5:]
    ch = {c: g[g.method == c].trade_fp_owner.sum() for c in CHANNELS}
    rows.append(dict(src=src, background=bg, runs=len(v6), pooled_fpr=fp.sum() / neg.sum(),
                     median_run_fpr=v6.fpr.median(), share_runs_any_fp=(fp > 0).mean(),
                     share_runs_fpr_gt_10pct=(v6.fpr > 0.10).mean(),
                     top5_share_of_fp=fp[top5].sum() / max(fp.sum(), 1),
                     pooled_fpr_without_top5=fp.drop(top5).sum() / neg.drop(top5).sum(),
                     **{f"fp_alone_{c}": n for c, n in ch.items()}, fp_v6=fp.sum()))
# fp_alone_<channel>: false positives if that channel alone were used at its own 1% threshold
# (not an attribution of v6's false positives; the channels overlap)
tab = pd.DataFrame(rows)
tab.to_csv(f"{OUT}/control_fpr.csv", index=False)

# ---- figure: share of control runs whose run-level FPR exceeds x (v6), small multiples
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"design seeds 0-199": "#2a78d6", "fresh seeds 1000-1099": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True, facecolor=SURF)
x = np.logspace(-3, 0, 400)
for ax, bg in zip(axes, BGS):
    ax.set_facecolor(SURF)
    pooled = []
    for src in SRC:
        f = d[(d.src == src) & (d.background == bg) & (d.method == "v6")].fpr.to_numpy()
        ax.step(x, [(f > t).mean() for t in x], where="post", lw=2, color=COLOR[src], label=src)
        pooled.append(tab[(tab.src == src) & (tab.background == bg)].pooled_fpr.iloc[0])
    ax.axvline(0.01, color=INK2, lw=1, ls=":")
    ax.text(0.97, 0.97, f"pooled FPR\ndesign {pooled[0]:.1%}\nfresh {pooled[1]:.1%}",
            transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 0.8)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(["0.1%", "1%", "10%", "100%"])
    ax.set_title(bg, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("run-level FPR")
axes[0].set_ylabel("share of control runs above x")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
fig.suptitle("v6 in control runs: false positives sit in a few runs (dotted: the 1% pooled target)",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f"{OUT}/control_fpr.png", dpi=180, facecolor=SURF)
pd.set_option("display.width", 250)
print(tab.round(4).to_string(index=False))
