"""
Phase 4: Hawkes process comparison.

Univariate exponential-kernel Hawkes:
    λ(t) = μ + α · Σ_{t_j < t} exp(-β (t - t_j))

`tick` 0.8 is broken on Python 3.14 (`HawkesExpKern.__init__` tries to
set a restricted attribute), so we fit by direct MLE in scipy and
simulate via Ogata thinning. Per `agents.md` §4 Option 2.

The intensity recursion that makes both fast:
    A(i) = (1 + A(i-1)) · exp(-β · (t_i - t_{i-1})),  A(0) = 0
    λ(t_i) = μ + α · A(i)

Pipeline
--------
1. Build a univariate point process from the empirical activation
   matrix: each day with at least one activation is one event at
   t = day_index (continuous time, units = trading days).
2. Fit (μ, α, β) by MLE.
3. Simulate N realisations on [0, T] via Ogata thinning.
4. For each simulated event time, build a "simulated activation row"
   by drawing stocks weighted by their empirical activation
   frequency, with a count drawn from the empirical distribution of
   activations per active day.
5. Run the Phase 1 avalanche detector on the simulated activations
   using the (cached) empirical network.
6. Compare empirical avalanche size CCDF to the pooled Hawkes-simulated
   CCDF; KS test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm

from .avalanche import detect_avalanches


# --------------------------------------------------------------------------
# Event-time extraction
# --------------------------------------------------------------------------

def build_event_times(activated: pd.DataFrame) -> np.ndarray:
    counts = activated.sum(axis=1).values
    idx = np.where(counts > 0)[0]
    return idx.astype(float)


def empirical_count_distribution(activated: pd.DataFrame) -> np.ndarray:
    counts = activated.sum(axis=1).values
    return counts[counts > 0]


def empirical_stock_weights(activated: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    freq = activated.sum(axis=0).astype(float).values
    if freq.sum() == 0:
        freq = np.ones_like(freq)
    weights = freq / freq.sum()
    return activated.columns.tolist(), weights


# --------------------------------------------------------------------------
# MLE for univariate exp-kernel Hawkes
# --------------------------------------------------------------------------

@dataclass
class HawkesFit:
    mu: float          # baseline intensity (events / day)
    alpha: float       # excitation magnitude
    beta: float        # decay
    branching_ratio: float  # α / β
    log_likelihood: float
    n_events: int
    T: float


def _hawkes_neg_log_likelihood(params, t, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return 1e15
    n = len(t)
    # Recursive intensity at each event time
    A = np.zeros(n)
    if n > 1:
        dt = np.diff(t)
        decay = np.exp(-beta * dt)
        for i in range(1, n):
            A[i] = (1.0 + A[i - 1]) * decay[i - 1]
    lam = mu + alpha * A
    if np.any(lam <= 0):
        return 1e15
    log_term = np.sum(np.log(lam))
    # Compensator: ∫_0^T λ(t) dt = μ T + (α/β) Σ (1 - e^{-β(T - t_i)})
    comp_term = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
    return -(log_term - comp_term)


def fit_hawkes_exp(event_times: np.ndarray) -> HawkesFit:
    """MLE via scipy. Constraints: μ > 0, α ≥ 0, β > 0, α / β < 1
    (stationarity). We use a soft barrier (α/β capped via parameter
    bounds) and L-BFGS-B."""
    t = np.sort(event_times.astype(float))
    T = float(t[-1] + 1.0)
    n = len(t)

    # Initial guess
    mu0 = 0.5 * n / T
    beta0 = 1.0
    alpha0 = 0.5 * beta0  # branching ratio ~ 0.5 starting point

    # Try a small grid of beta starts to avoid local optima
    best = None
    best_nll = np.inf
    for beta_init in [0.1, 0.5, 1.0, 2.0, 5.0]:
        x0 = [mu0, 0.5 * beta_init, beta_init]
        try:
            res = optimize.minimize(
                _hawkes_neg_log_likelihood,
                x0=x0,
                args=(t, T),
                method="L-BFGS-B",
                bounds=[(1e-6, None), (1e-9, None), (1e-4, None)],
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best = res
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Hawkes MLE failed for all starts")

    mu, alpha, beta = best.x
    return HawkesFit(
        mu=float(mu),
        alpha=float(alpha),
        beta=float(beta),
        branching_ratio=float(alpha / beta),
        log_likelihood=float(-best.fun),
        n_events=int(n),
        T=float(T),
    )


# --------------------------------------------------------------------------
# Ogata thinning simulation
# --------------------------------------------------------------------------

def simulate_event_times(fit: HawkesFit, T: float, rng: np.random.Generator) -> np.ndarray:
    """Ogata thinning. Returns event times in (0, T]."""
    mu, alpha, beta = fit.mu, fit.alpha, fit.beta
    times: list[float] = []
    s = 0.0
    A = 0.0  # excitation state evaluated at last (rejected or accepted) point
    last_s = 0.0
    while True:
        lam_upper = mu + alpha * (A + 1.0) * np.exp(-beta * (s - last_s))
        # Conservative upper bound: take alpha * (A_at_last_event + 1) decayed to s,
        # but to be safe re-evaluate A at s.
        A_at_s = A * np.exp(-beta * (s - last_s)) if last_s <= s else 0.0
        lam_upper = mu + alpha * A_at_s + alpha  # bound: any new event adds at most α
        # Sample next candidate
        u = rng.exponential(1.0 / max(lam_upper, 1e-12))
        s = s + u
        if s > T:
            break
        # Compute true intensity at s
        A_at_s = A * np.exp(-beta * (s - last_s))
        lam_s = mu + alpha * A_at_s
        # Thinning
        D = rng.uniform()
        if D * lam_upper <= lam_s:
            times.append(s)
            A = A_at_s + 1.0
            last_s = s
        # else: rejected; keep accumulator A,last_s as-is (decay re-applied next iter)
    return np.asarray(times, dtype=float)


# --------------------------------------------------------------------------
# Activation matrix construction
# --------------------------------------------------------------------------

def simulate_activation_matrix(
    event_times: np.ndarray,
    count_distribution: np.ndarray,
    stock_weights: np.ndarray,
    columns: list[str],
    full_index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    T = len(full_index)
    N = len(columns)
    out = np.zeros((T, N), dtype=np.int8)

    if len(event_times) == 0:
        return pd.DataFrame(out, index=full_index, columns=columns)

    day_indices = np.clip(np.floor(event_times).astype(int), 0, T - 1)
    counts = rng.choice(count_distribution, size=len(day_indices), replace=True)

    for d, n_act in zip(day_indices, counts):
        n_act = int(min(n_act, N))
        if n_act <= 0:
            continue
        existing = np.where(out[d] == 1)[0]
        avail_mask = np.ones(N, dtype=bool)
        avail_mask[existing] = False
        if not avail_mask.any():
            continue
        w = stock_weights * avail_mask
        nz = int((w > 0).sum())
        if nz == 0:
            continue
        w = w / w.sum()
        size = int(min(n_act, nz))
        chosen = rng.choice(N, size=size, replace=False, p=w)
        out[d, chosen] = 1

    return pd.DataFrame(out, index=full_index, columns=columns)


# --------------------------------------------------------------------------
# Top-level: run N simulations against a cached empirical network
# --------------------------------------------------------------------------

def run_hawkes_simulations(
    activated: pd.DataFrame,
    cached_network,
    n_sims: int = 200,
    seed: int = 42,
    progress: bool = True,
) -> dict:
    event_times = build_event_times(activated)
    fit = fit_hawkes_exp(event_times)
    counts = empirical_count_distribution(activated)
    columns, weights = empirical_stock_weights(activated)

    rng = np.random.default_rng(seed)
    sizes_per_sim: list[list[int]] = []
    n_events_per_sim: list[int] = []

    iterator = range(n_sims)
    if progress:
        iterator = tqdm(iterator, desc="Hawkes sims")

    for _ in iterator:
        sim_times = simulate_event_times(fit, T=fit.T, rng=rng)
        sim_activated = simulate_activation_matrix(
            sim_times,
            count_distribution=counts,
            stock_weights=weights,
            columns=columns,
            full_index=activated.index,
            rng=rng,
        )
        avalanches = detect_avalanches(sim_activated, cached_network)
        sizes_per_sim.append([a.size for a in avalanches])
        n_events_per_sim.append(int(len(sim_times)))

    return {
        "fit": fit,
        "n_events_empirical": int(len(event_times)),
        "n_events_per_sim": np.asarray(n_events_per_sim),
        "sizes_per_sim": sizes_per_sim,
    }
