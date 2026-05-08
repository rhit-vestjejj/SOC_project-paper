"""
Phase 5.2: Crackling-noise size-duration scaling.

Sethna et al. (2001): in self-organized critical systems the typical
avalanche size at a given duration scales as ⟨S | T⟩ ~ T^γ, where
γ depends only on the universality class. Mean-field SOC predicts
γ ≈ 2; directed percolation γ ≈ 1.78.

Lognormal processes do not produce such a clean power-law scaling
relationship between size and duration — they can produce heavy-tailed
marginal distributions but cannot mimic the joint S-T scaling.

Estimator: bin avalanches by integer duration T, take mean ln(size)
within each well-populated bin, regress on ln(T).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ScalingFit:
    label: str
    gamma: float
    gamma_se: float
    intercept: float
    n_bins: int
    n_avalanches_total: int
    n_avalanches_used: int
    binned: pd.DataFrame  # columns: T, mean_log_size, count


def size_duration_scaling(
    avalanches_df: pd.DataFrame,
    label: str = "",
    min_count_per_bin: int = 20,
    min_duration: int = 1,
) -> ScalingFit:
    """
    Bin avalanches by integer duration T (>= min_duration), take mean
    log size per bin (drop bins with fewer than `min_count_per_bin`
    points), and OLS-regress mean_log_size ~ log(T).

    Returns a ScalingFit with γ = slope, its standard error, the
    binned data, and bookkeeping.
    """
    df = avalanches_df[["size", "duration"]].dropna()
    df = df[(df["size"] > 0) & (df["duration"] >= min_duration)]
    n_total = len(df)

    if n_total == 0:
        return ScalingFit(label, np.nan, np.nan, np.nan, 0, 0, 0, pd.DataFrame())

    df["log_size"] = np.log(df["size"].astype(float))
    grouped = df.groupby("duration")["log_size"].agg(["mean", "count"]).reset_index()
    grouped = grouped.rename(columns={"mean": "mean_log_size", "count": "count"})
    grouped = grouped[grouped["count"] >= min_count_per_bin]

    n_bins = len(grouped)
    n_used = int(grouped["count"].sum()) if n_bins else 0

    if n_bins < 3:
        return ScalingFit(label, np.nan, np.nan, np.nan, n_bins, n_total, n_used, grouped)

    x = np.log(grouped["duration"].astype(float).values)
    y = grouped["mean_log_size"].values
    res = stats.linregress(x, y)

    return ScalingFit(
        label=label,
        gamma=float(res.slope),
        gamma_se=float(res.stderr),
        intercept=float(res.intercept),
        n_bins=n_bins,
        n_avalanches_total=n_total,
        n_avalanches_used=n_used,
        binned=grouped.assign(log_T=x, log_S=y),
    )
