"""
Phase 2: Rolling branching ratio as an early-warning indicator.

For each trading day t, compute

    sigma_W(t) = sum(secondary) / sum(primary)

over avalanches whose start_day falls in (t - W, t]. Implemented as a
groupby-then-rolling-sum on the per-avalanche records, so it is O(T + N)
and reuses the Phase 1 outputs without re-running detection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# --- Crisis events (per agents.md §2.2) -----------------------------------
# Dates pinned to the trading-day on or just after the canonical event.
CRISIS_EVENTS: list[tuple[str, str]] = [
    ("Lehman",       "2008-09-15"),
    ("Eurozone/US downgrade", "2011-08-05"),
    ("China selloff", "2015-08-24"),
    ("Volmageddon",  "2018-02-05"),
    ("COVID",        "2020-03-09"),
    ("Fed liftoff",  "2022-01-05"),
]


def rolling_branching_ratio(
    avalanches: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    window: int,
) -> pd.Series:
    """
    Parameters
    ----------
    avalanches : DataFrame with columns ['start_day', 'primary', 'secondary'].
    trading_days : full reference index (used for reindexing).
    window : trailing window length in trading days.
    """
    a = avalanches.copy()
    a["start_day"] = pd.to_datetime(a["start_day"])
    daily = (
        a.groupby("start_day")[["primary", "secondary"]]
        .sum()
        .reindex(trading_days, fill_value=0)
        .sort_index()
    )
    rolled = daily.rolling(window=window, min_periods=window).sum()
    sigma = rolled["secondary"] / rolled["primary"].replace(0, np.nan)
    sigma.name = f"sigma_{window}"
    return sigma


def market_log_returns(market_prices: pd.DataFrame, col: str = "GSPC") -> pd.Series:
    p = market_prices[col].astype(float)
    return np.log(p / p.shift(1)).dropna()


def forward_realized_vol(returns: pd.Series, horizon: int) -> pd.Series:
    """At label t: std of returns over the next `horizon` trading days
    [t+1 ... t+horizon], annualized."""
    rolled = returns.rolling(window=horizon, min_periods=horizon).std()
    fwd = rolled.shift(-horizon)
    fwd.name = f"fwd_vol_{horizon}"
    return fwd * np.sqrt(252)


def forward_max_drawdown(prices: pd.Series, horizon: int) -> pd.Series:
    """Max drawdown over the *next* `horizon` trading days, expressed as a
    positive number (so larger = worse)."""
    log_p = np.log(prices.astype(float))
    out = pd.Series(np.nan, index=prices.index)
    arr = log_p.values
    for i in range(len(arr) - horizon):
        window = arr[i + 1 : i + 1 + horizon]
        running_max = np.maximum.accumulate(window)
        dd = running_max - window  # in log space; ~ -log(1-dd)
        out.iloc[i] = float(dd.max())
    out.name = f"max_drawdown_{horizon}"
    return out


@dataclass
class CorrelationResult:
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float


def lead_lag_correlation(sigma: pd.Series, future: pd.Series) -> CorrelationResult:
    df = pd.concat([sigma, future], axis=1).dropna()
    if len(df) < 30:
        return CorrelationResult(len(df), np.nan, np.nan, np.nan, np.nan)
    pr = stats.pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    sr = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return CorrelationResult(
        n=len(df),
        pearson_r=float(pr.statistic),
        pearson_p=float(pr.pvalue),
        spearman_r=float(sr.statistic),
        spearman_p=float(sr.pvalue),
    )


@dataclass
class IncrementalRegressionResult:
    horizon: int
    n: int
    r2_baseline: float       # fwd_vol ~ lagged_vol
    r2_full: float           # fwd_vol ~ lagged_vol + sigma
    delta_r2: float
    coef_sigma: float
    se_sigma: float
    t_sigma: float
    p_sigma: float           # HAC, two-sided
    coef_lagged_vol: float
    p_lagged_vol: float


def incremental_predictive_regression(
    sigma: pd.Series,
    market_returns: pd.Series,
    horizon: int,
) -> IncrementalRegressionResult:
    """
    Test whether sigma_W(t) has predictive power for forward realized vol
    *after* controlling for trailing realized vol (which is the obvious
    confounder, given vol persistence).

    Newey-West HAC standard errors with `maxlags = horizon` to handle the
    overlap induced by horizon-day forward and backward windows.
    """
    fwd = forward_realized_vol(market_returns, horizon=horizon)
    lagged = (
        market_returns.rolling(window=horizon, min_periods=horizon).std()
        * np.sqrt(252)
    )
    lagged.name = "lagged_vol"

    df = pd.concat([fwd.rename("fwd_vol"), lagged, sigma.rename("sigma")], axis=1).dropna()
    n = len(df)

    # Baseline: vol-only
    X0 = sm.add_constant(df[["lagged_vol"]])
    m0 = sm.OLS(df["fwd_vol"], X0).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon}
    )
    # Full: vol + sigma
    X1 = sm.add_constant(df[["lagged_vol", "sigma"]])
    m1 = sm.OLS(df["fwd_vol"], X1).fit(
        cov_type="HAC", cov_kwds={"maxlags": horizon}
    )

    return IncrementalRegressionResult(
        horizon=horizon,
        n=n,
        r2_baseline=float(m0.rsquared),
        r2_full=float(m1.rsquared),
        delta_r2=float(m1.rsquared - m0.rsquared),
        coef_sigma=float(m1.params["sigma"]),
        se_sigma=float(m1.bse["sigma"]),
        t_sigma=float(m1.tvalues["sigma"]),
        p_sigma=float(m1.pvalues["sigma"]),
        coef_lagged_vol=float(m1.params["lagged_vol"]),
        p_lagged_vol=float(m1.pvalues["lagged_vol"]),
    )


def precrisis_test(
    sigma: pd.Series,
    crisis_date: pd.Timestamp,
    lookback: int = 60,
) -> dict:
    """Mann-Whitney U (one-sided): is sigma in the `lookback` days before
    `crisis_date` greater than sigma over the rest of the sample?"""
    sigma = sigma.dropna()
    crisis_date = pd.Timestamp(crisis_date)
    end = sigma.index.searchsorted(crisis_date)
    start = max(0, end - lookback)
    pre = sigma.iloc[start:end]
    rest = sigma.drop(sigma.index[start:end])
    if len(pre) < 10 or len(rest) < 30:
        return {
            "n_pre": len(pre), "n_rest": len(rest),
            "median_pre": float(pre.median()) if len(pre) else np.nan,
            "median_rest": float(rest.median()) if len(rest) else np.nan,
            "U": np.nan, "p_one_sided": np.nan,
        }
    U, p = stats.mannwhitneyu(pre, rest, alternative="greater")
    return {
        "n_pre": int(len(pre)),
        "n_rest": int(len(rest)),
        "median_pre": float(pre.median()),
        "median_rest": float(rest.median()),
        "U": float(U),
        "p_one_sided": float(p),
    }
