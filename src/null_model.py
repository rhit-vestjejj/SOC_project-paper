"""
Phase 1.8: Shuffled null model.

For each stock independently, randomly shuffle the time indices of its
residual return series. Preserves marginal distributions but destroys
temporal structure and cross-stock causality.

Re-runs activation -> network -> avalanche detection -> sizes for each
shuffle, and returns the distribution of avalanche sizes across shuffles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .activation import detect_activations
from .avalanche import detect_avalanches
from .network import build_correlation_provider


def _shuffle_columns_independently(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle each column's values independently, preserving NaN positions
    by shuffling only the non-NaN entries."""
    out = df.copy()
    for col in out.columns:
        s = np.asarray(out[col].values, dtype=float).copy()
        mask = ~np.isnan(s)
        if mask.sum() < 2:
            continue
        vals = s[mask].copy()
        rng.shuffle(vals)
        s[mask] = vals
        out[col] = s
    return out


def run_null_model(
    residuals: pd.DataFrame,
    k: float,
    tau: float,
    window: int,
    n_shuffles: int = 100,
    seed: int = 42,
    rebuild_every: int = 5,
    progress: bool = True,
    same_day_propagation: bool = False,
) -> dict:
    """
    Returns
    -------
    dict with:
        sizes_per_shuffle : list[list[int]]
        durations_per_shuffle : list[list[int]]
        n_avalanches_per_shuffle : np.ndarray
    """
    rng = np.random.default_rng(seed)
    sizes_runs: list[list[int]] = []
    durs_runs: list[list[int]] = []
    counts: list[int] = []

    iterator = range(n_shuffles)
    if progress:
        iterator = tqdm(iterator, desc="null shuffles")

    for _ in iterator:
        shuffled = _shuffle_columns_independently(residuals, rng)
        activated = detect_activations(shuffled, k=k, window=window)
        net = build_correlation_provider(
            shuffled, tau=tau, window=window, rebuild_every=rebuild_every
        )
        avalanches = detect_avalanches(
            activated, net, same_day_propagation=same_day_propagation
        )
        sizes_runs.append([a.size for a in avalanches])
        durs_runs.append([a.duration for a in avalanches])
        counts.append(len(avalanches))

    return {
        "sizes_per_shuffle": sizes_runs,
        "durations_per_shuffle": durs_runs,
        "n_avalanches_per_shuffle": np.array(counts),
    }


def pooled_quantiles(runs: list[list[float]], q=(0.025, 0.5, 0.975)) -> pd.DataFrame:
    """For diagnostic purposes: per-rank quantiles of sorted sizes across runs."""
    if not runs:
        return pd.DataFrame()
    max_len = max(len(r) for r in runs)
    sorted_runs = []
    for r in runs:
        a = np.sort(np.asarray(r, dtype=float))[::-1]  # decreasing
        if len(a) < max_len:
            a = np.pad(a, (0, max_len - len(a)), constant_values=np.nan)
        sorted_runs.append(a)
    M = np.vstack(sorted_runs)
    qs = np.nanquantile(M, q, axis=0)
    return pd.DataFrame(qs.T, columns=[f"q{int(qi*1000)/10}" for qi in q])
