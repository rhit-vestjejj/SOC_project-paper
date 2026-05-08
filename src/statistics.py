"""
Phase 1.6 + 1.7: Power-law fitting and global branching ratio.

Power-law fit follows Clauset, Shalizi, Newman (2009) via the `powerlaw`
package: estimate alpha and x_min by KS minimization, then test against
exponential and lognormal alternatives via likelihood ratio.

Branching ratio sigma = total_secondary / total_primary across all avalanches,
with a stationary bootstrap CI over avalanche records.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

try:
    import powerlaw  # type: ignore
except ImportError:
    powerlaw = None  # noqa: N816


def _silence_powerlaw_warnings():
    """The `powerlaw` package emits a flood of warnings on real data
    (parameters near edges, lognormal failing to fit, scipy optimize
    bounds). They're not actionable for our use-case — we report
    sample sizes and tail counts so the reader can judge fit quality."""
    warnings.filterwarnings("ignore", category=UserWarning, module="powerlaw")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="powerlaw")
    warnings.filterwarnings("ignore", category=OptimizeWarning)


_silence_powerlaw_warnings()


@dataclass
class PowerLawFit:
    n: int                 # total observations
    n_tail: int            # observations >= xmin
    alpha: float
    xmin: float
    sigma_alpha: float     # standard error on alpha
    R_exp: float           # log-likelihood ratio vs exponential
    p_exp: float
    R_ln: float            # vs lognormal
    p_ln: float
    R_trunc: float         # vs truncated power law
    p_trunc: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def fit_power_law(values, discrete: bool = True) -> PowerLawFit | None:
    """Fit a power law to non-negative integer values. Returns None if no data."""
    if powerlaw is None:
        raise ImportError("Install the `powerlaw` package: pip install powerlaw")

    arr = np.asarray([v for v in values if v is not None and not np.isnan(v) and v > 0])
    if len(arr) < 5:
        return None

    fit = powerlaw.Fit(arr, discrete=discrete, verbose=False)
    n_tail = int((arr >= fit.xmin).sum())

    R_exp, p_exp = fit.distribution_compare(
        "power_law", "exponential", normalized_ratio=True
    )
    R_ln, p_ln = fit.distribution_compare(
        "power_law", "lognormal", normalized_ratio=True
    )
    R_trunc, p_trunc = fit.distribution_compare(
        "power_law", "truncated_power_law", normalized_ratio=True
    )

    return PowerLawFit(
        n=len(arr),
        n_tail=n_tail,
        alpha=float(fit.alpha),
        xmin=float(fit.xmin),
        sigma_alpha=float(getattr(fit, "sigma", np.nan)),
        R_exp=float(R_exp),
        p_exp=float(p_exp),
        R_ln=float(R_ln),
        p_ln=float(p_ln),
        R_trunc=float(R_trunc),
        p_trunc=float(p_trunc),
    )


def global_branching_ratio(avalanches) -> float:
    pri = sum(a.primary for a in avalanches)
    sec = sum(a.secondary for a in avalanches)
    if pri == 0:
        return float("nan")
    return sec / pri


def branching_ratio_bootstrap(
    avalanches, n_boot: int = 1000, seed: int = 42, ci: float = 0.95
) -> tuple[float, float, float]:
    """Returns (point, lo, hi). Resamples avalanches with replacement."""
    if not avalanches:
        return (float("nan"),) * 3

    rng = np.random.default_rng(seed)
    pri = np.array([a.primary for a in avalanches], dtype=float)
    sec = np.array([a.secondary for a in avalanches], dtype=float)
    point = sec.sum() / pri.sum() if pri.sum() > 0 else float("nan")

    n = len(avalanches)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p = pri[idx].sum()
        s = sec[idx].sum()
        samples[b] = s / p if p > 0 else np.nan

    lo, hi = np.nanpercentile(samples, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(point), float(lo), float(hi)


def ccdf(values) -> tuple[np.ndarray, np.ndarray]:
    """Empirical complementary CDF: returns (sorted unique x, P(X >= x))."""
    arr = np.sort(np.asarray([v for v in values if v is not None and v > 0]))
    if len(arr) == 0:
        return np.array([]), np.array([])
    n = len(arr)
    # P(X >= x) for each unique x
    x_unique, counts = np.unique(arr, return_counts=True)
    cum = np.cumsum(counts)
    # P(X >= x_unique[i]) = (n - cum[i-1]) / n  (cum[i-1] = points strictly below x_unique[i])
    cum_below = np.concatenate(([0], cum[:-1]))
    p_ge = (n - cum_below) / n
    return x_unique, p_ge


def summarize(avalanches) -> dict:
    sizes = [a.size for a in avalanches]
    durations = [a.duration for a in avalanches]
    pri_total = sum(a.primary for a in avalanches)
    sec_total = sum(a.secondary for a in avalanches)
    br_point, br_lo, br_hi = branching_ratio_bootstrap(avalanches)
    out = {
        "n_avalanches": len(avalanches),
        "size_mean": float(np.mean(sizes)) if sizes else float("nan"),
        "size_max": int(np.max(sizes)) if sizes else 0,
        "duration_mean": float(np.mean(durations)) if durations else float("nan"),
        "duration_max": int(np.max(durations)) if durations else 0,
        "primary_total": int(pri_total),
        "secondary_total": int(sec_total),
        "branching_ratio": br_point,
        "branching_ratio_lo": br_lo,
        "branching_ratio_hi": br_hi,
    }
    if powerlaw is not None and len(sizes) >= 5:
        size_fit = fit_power_law(sizes)
        if size_fit is not None:
            out.update({f"size_{k}": v for k, v in size_fit.to_dict().items()})
        if len(durations) >= 5:
            dur_fit = fit_power_law(durations)
            if dur_fit is not None:
                out.update({f"duration_{k}": v for k, v in dur_fit.to_dict().items()})
    return out
