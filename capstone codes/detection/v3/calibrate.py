r"""calibrate.py -- control-calibrated thresholds, and every run scored at them.

Algorithm 1's fixed theta = 0.9 flags 40-45% of honest volume in the active backgrounds
(detection v2). A fair comparison across backgrounds fixes the false-positive rate instead:

    theta_cal(background, config) = the smallest theta on the curve grid (0.50, 0.51, ..., 1.00)
                                    whose POOLED trade FPR over the background's control runs
                                    is <= target (default 1%)

    pooled FPR(theta) = sum_runs false-positive volume(theta) / sum_runs honest volume

If no grid value reaches the target, theta_cal = inf (nothing is flagged). Control runs come from
the ``scenarios`` sweep (scenario "control"; all five backgrounds, same seeds as every other sweep:
the evasive sweep's control cells are the same runs).

Every run of every sweep is then scored at its background's theta_cal, from its threshold curve:
trade recall / precision / FPR (volume), the same-group recall, and wallet recall / FPR.
"""
import os

import numpy as np
import pandas as pd

from washtrade.v7 import design as D
from washtrade.v7.generate import dataset_root

TARGET_FPR = 0.01
OUT_NAME = "detection_v3"


def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def control_thresholds(curves: pd.DataFrame, target=TARGET_FPR) -> pd.DataFrame:
    """curves of control runs -> one row per (background, config): theta_cal, its pooled FPR,
    the number of control runs, and the pooled FPR at the paper's theta = 0.9."""
    rows = []
    for (bg, cfg), c in curves.groupby(["background", "config"]):
        pooled = c.groupby("theta")[["trade_fp", "trade_neg"]].sum()
        fpr = pd.Series(_div(pooled["trade_fp"], pooled["trade_neg"]), index=pooled.index)
        ok = fpr[fpr <= target]
        theta = float(ok.index.min()) if len(ok) else float("inf")
        at09 = fpr[np.isclose(fpr.index, 0.9)]
        rows.append(dict(background=bg, config=cfg, theta_cal=theta,
                         fpr_at_cal=(float(fpr.loc[theta]) if np.isfinite(theta) else 0.0),
                         fpr_at_0_9=(float(at09.iloc[0]) if len(at09) else np.nan),
                         n_control_runs=int(c["run_id"].nunique()), target_fpr=target))
    return pd.DataFrame(rows)


def score_at(curves: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, config): metrics at the background's calibrated threshold."""
    c = curves.merge(thresholds[["background", "config", "theta_cal"]],
                     on=["background", "config"], how="inner")
    finite = np.isfinite(c["theta_cal"])
    at = c[finite & np.isclose(c["theta"], c["theta_cal"].where(finite, -1))].copy()
    # theta_cal = inf: nothing flagged -> take the theta = 1.0 row and zero its positives
    none = c[~finite & np.isclose(c["theta"], 1.0)].copy()
    for col in ("trade_tp", "trade_fp", "trade_tp_same", "wallet_tp", "wallet_fp"):
        none[col] = 0
    at = pd.concat([at, none], ignore_index=True)
    at["trade_recall"] = _div(at["trade_tp"], at["trade_pos"])
    at["trade_precision"] = _div(at["trade_tp"], at["trade_tp"] + at["trade_fp"])
    at["trade_fpr"] = _div(at["trade_fp"], at["trade_neg"])
    at["trade_recall_same"] = _div(at["trade_tp_same"], at["trade_pos_same"])
    at["wallet_recall"] = _div(at["wallet_tp"], at["wallet_pos"])
    at["wallet_fpr"] = _div(at["wallet_fp"], at["wallet_neg"])
    return at.drop(columns=["theta"]).reset_index(drop=True)


def _curves(sweep, data):
    p = os.path.join(dataset_root(sweep, out_dir=data), OUT_NAME, "curves.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def calibrate_all(data, target=TARGET_FPR, verbose=True):
    """Thresholds from the scenarios sweep's control runs, then calibrated.csv for every sweep
    whose curves exist. Returns the thresholds table."""
    sc = _curves("scenarios", data)
    if sc is None:
        raise FileNotFoundError("calibration needs the scenarios sweep's curves (control runs)")
    ctrl = sc[sc["f_scenario"] == "control"]
    th = control_thresholds(ctrl, target)
    th.to_csv(os.path.join(data, f"{OUT_NAME}_thresholds.csv"), index=False)
    if verbose:
        print(th.round(4).to_string(index=False))
    for sw in D.SWEEPS:
        cur = _curves(sw, data)
        if cur is None:
            continue
        out = score_at(cur, th)
        out.to_csv(os.path.join(dataset_root(sw, out_dir=data), OUT_NAME, "calibrated.csv"),
                   index=False)
        if verbose:
            print(f"{sw}: {out['run_id'].nunique()} runs scored at the calibrated thresholds")
    return th
