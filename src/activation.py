"""
Phase 1.3: Activation detection.

Stock i is activated on day t iff
    |residual_{i,t}| > k * sigma_{i,t}
where sigma_{i,t} is the trailing W-day rolling std of that stock's residuals.

Returns a 0/1 DataFrame with the same shape as the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_K, ROLLING_WINDOW


def detect_activations(
    series: pd.DataFrame,
    k: float = DEFAULT_K,
    window: int = ROLLING_WINDOW,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    series : DataFrame, shape (T, N)
        Residuals (or raw returns).
    k : float
        Threshold multiplier.
    window : int
        Trailing window for the rolling std.
    min_periods : int, optional
        Minimum observations required to compute the std. Defaults to window.

    Returns
    -------
    DataFrame of 0/1 with same shape, dtype int8. NaNs are treated as 0.
    """
    if min_periods is None:
        min_periods = window

    # Use shifted std so we don't peek at today's value when judging today.
    sigma = (
        series.rolling(window=window, min_periods=min_periods)
        .std()
        .shift(1)
    )

    activated = (series.abs() > k * sigma).astype("int8")
    activated = activated.where(~series.isna(), 0).astype("int8")
    return activated


def activation_summary(activated: pd.DataFrame) -> pd.Series:
    """Per-day count of activated stocks."""
    return activated.sum(axis=1)
