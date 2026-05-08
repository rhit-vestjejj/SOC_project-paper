"""
Phase 4 driver: Hawkes process comparison.

Fits a univariate exponential Hawkes process to the empirical sequence
of "active days" (days with >= 1 stock activated), simulates N paths
on the same horizon, attaches stock identities by drawing from
empirical activation frequencies, runs avalanche detection on the
simulated activations using the (cached) empirical network, and
compares avalanche size distributions.

Usage:
    python run_phase4.py                     # 200 sims, default
    python run_phase4.py --n-sims 50         # quick
    python run_phase4.py --n-sims 1000       # paper-grade
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.activation import detect_activations
from src.config import (
    DATA_DIR,
    DEFAULT_K,
    DEFAULT_TAU,
    FIG_DIR,
    RANDOM_SEED,
    ROLLING_WINDOW,
    TAB_DIR,
)
from src.hawkes import run_hawkes_simulations
from src.network import build_cached_network, build_correlation_provider
from src.plotting import plot_ccdf_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=200)
    parser.add_argument("--rebuild-every", type=int, default=5)
    parser.add_argument("--cache-network-from-disk", action="store_true",
                        help="reuse cached network pickle if present")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    print("== Loading residuals + computing empirical activations ==")
    residuals = pd.read_parquet(DATA_DIR / "residuals.parquet")
    activated = detect_activations(residuals, k=DEFAULT_K, window=ROLLING_WINDOW)
    print(f"  residuals: {residuals.shape}, activated: {activated.shape}")
    n_active_days = int((activated.sum(axis=1) > 0).sum())
    print(f"  active days: {n_active_days:,} of {len(activated):,}")

    # ------------------------------------------------------------------
    # Build (and cache) empirical network
    # ------------------------------------------------------------------
    cache_path = DATA_DIR / "cached_network.pkl"
    if args.cache_network_from_disk and cache_path.exists():
        print(f"\n== Loading cached empirical network from {cache_path.name} ==")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
    else:
        print("\n== Building (and caching) empirical network ==")
        provider = build_correlation_provider(
            residuals, tau=DEFAULT_TAU, window=ROLLING_WINDOW,
            rebuild_every=args.rebuild_every,
        )
        cached = build_cached_network(provider, dates=residuals.index)
        with open(cache_path, "wb") as f:
            pickle.dump(cached, f)
        print(f"  cached → {cache_path}")
    n_unique_snapshots = len({id(v) for v in cached.snapshots.values()})
    print(f"  {n_unique_snapshots} unique adjacency snapshots cached")

    # ------------------------------------------------------------------
    # Hawkes fit + simulation
    # ------------------------------------------------------------------
    print(f"\n== Hawkes fit + {args.n_sims} simulations ==")
    out = run_hawkes_simulations(
        activated,
        cached,
        n_sims=args.n_sims,
        seed=RANDOM_SEED,
    )

    fit = out["fit"]
    print(
        f"\n  Hawkes fit: μ={fit.mu:.4f}/day  α={fit.alpha:.4f}  β={fit.beta:.4f}  "
        f"branching_ratio={fit.branching_ratio:.4f}  n_events={fit.n_events}"
    )
    print(f"  empirical events: {out['n_events_empirical']}")
    print(f"  simulated events per path: mean={out['n_events_per_sim'].mean():.1f}  "
          f"sd={out['n_events_per_sim'].std():.1f}")

    # ------------------------------------------------------------------
    # Empirical avalanche sizes (load from Phase 1 outputs)
    # ------------------------------------------------------------------
    emp_df = pd.read_csv(TAB_DIR / "avalanches_residuals.csv")
    empirical_sizes = emp_df["size"].tolist()

    # ------------------------------------------------------------------
    # Persist sims (so plot rerun is cheap)
    # ------------------------------------------------------------------
    sim_blob_path = DATA_DIR / "hawkes_sim_sizes.pkl"
    with open(sim_blob_path, "wb") as f:
        pickle.dump({
            "fit": fit.__dict__,
            "sizes_per_sim": out["sizes_per_sim"],
            "n_events_per_sim": out["n_events_per_sim"].tolist(),
            "empirical_sizes": empirical_sizes,
        }, f)

    # ------------------------------------------------------------------
    # KS test: empirical vs pooled simulated
    # ------------------------------------------------------------------
    pooled_sim_sizes = [s for sizes in out["sizes_per_sim"] for s in sizes]
    if not pooled_sim_sizes:
        print("\n  ! WARNING: no simulated avalanches; check Hawkes fit")
        ks_stat, ks_p = float("nan"), float("nan")
    else:
        ks = stats.ks_2samp(empirical_sizes, pooled_sim_sizes)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    print(
        f"\n  KS test (empirical vs pooled Hawkes sims): "
        f"D={ks_stat:.4f}, p={ks_p:.2e}"
    )

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    sim_max_per_path = np.array([max(s) if s else 0 for s in out["sizes_per_sim"]])
    sim_n_per_path = np.array([len(s) for s in out["sizes_per_sim"]])
    emp_max = max(empirical_sizes) if empirical_sizes else 0
    emp_n = len(empirical_sizes)

    p_value_max = float((sim_max_per_path >= emp_max).mean())

    summary = {
        "n_sims": args.n_sims,
        "hawkes": {
            "mu": fit.mu,
            "alpha": fit.alpha,
            "beta": fit.beta,
            "branching_ratio": fit.branching_ratio,
            "n_events_empirical": fit.n_events,
            "T_days": fit.T,
        },
        "n_avalanches_empirical": emp_n,
        "n_avalanches_per_sim_mean": float(sim_n_per_path.mean()),
        "n_avalanches_per_sim_sd": float(sim_n_per_path.std()),
        "max_size_empirical": int(emp_max),
        "max_size_per_sim_mean": float(sim_max_per_path.mean()),
        "max_size_per_sim_p95": float(np.percentile(sim_max_per_path, 95)),
        "p_value_max_size": p_value_max,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
    }
    print("\n== Summary ==")
    print(f"  empirical: n_avalanches={emp_n:,}, max_size={emp_max}")
    print(f"  Hawkes sims: n_avalanches/sim mean={summary['n_avalanches_per_sim_mean']:.1f}, "
          f"max_size mean={summary['max_size_per_sim_mean']:.1f}, "
          f"p95={summary['max_size_per_sim_p95']:.1f}")
    print(f"  p_value (max_size_emp vs sim distribution): {p_value_max:.4f}")

    with open(TAB_DIR / "phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_ccdf_comparison(
        {
            "empirical (Phase 1)": empirical_sizes,
            "Hawkes-simulated (pooled)": pooled_sim_sizes,
        },
        title="Avalanche size CCDF: empirical vs Hawkes-simulated",
        xlabel="avalanche size (stocks)",
        out_path=FIG_DIR / "ccdf_empirical_vs_hawkes.png",
    )

    print("\n== Done ==")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TAB_DIR}")


if __name__ == "__main__":
    main()
