"""
End-to-end Phase 1 driver.

Usage:
    python run_baseline.py                    # full pipeline, default params
    python run_baseline.py --skip-data        # use cached data
    python run_baseline.py --quick            # subset of params, fewer null shuffles
    python run_baseline.py --no-null          # skip null model
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.activation import detect_activations
from src.avalanche import avalanches_to_frame, detect_avalanches
from src.config import (
    DATA_DIR,
    DEFAULT_K,
    DEFAULT_TAU,
    FIG_DIR,
    K_GRID,
    RANDOM_SEED,
    ROLLING_WINDOW,
    TAB_DIR,
    TAU_GRID,
)
from src.data_acquisition import build_dataset, compute_log_returns
from src.network import build_correlation_provider
from src.null_model import run_null_model
from src.plotting import plot_ccdf, plot_empirical_vs_null
from src.statistics import summarize


def load_or_build_data(skip_data: bool):
    if skip_data and (DATA_DIR / "residuals.parquet").exists():
        residuals = pd.read_parquet(DATA_DIR / "residuals.parquet")
        if (DATA_DIR / "log_returns.parquet").exists():
            log_returns = pd.read_parquet(DATA_DIR / "log_returns.parquet")
        else:
            prices = pd.read_parquet(DATA_DIR / "raw_prices.parquet")
            log_returns = compute_log_returns(prices)
        return log_returns, residuals
    bundle = build_dataset()
    return bundle["log_returns"], bundle["residuals"]


def run_one(
    series: pd.DataFrame,
    label: str,
    k: float,
    tau: float,
    rebuild_every: int,
):
    activated = detect_activations(series, k=k, window=ROLLING_WINDOW)
    net = build_correlation_provider(
        series, tau=tau, window=ROLLING_WINDOW, rebuild_every=rebuild_every
    )
    avalanches = detect_avalanches(activated, net)
    summary = summarize(avalanches)
    summary.update({"label": label, "k": k, "tau": tau})
    return avalanches, summary, activated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-null", action="store_true")
    parser.add_argument("--null-from-cache", action="store_true",
                        help="reuse pickled null shuffles in data/null_shuffles.pkl")
    parser.add_argument("--rebuild-every", type=int, default=5,
                        help="recompute correlation network every N trading days "
                             "(1 = strict baseline, larger = faster)")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    print("== Loading data ==")
    log_returns, residuals = load_or_build_data(args.skip_data)
    print(f"log_returns shape: {log_returns.shape}")
    print(f"residuals shape:   {residuals.shape}")

    # ------------------------------------------------------------------
    # Baseline (k=2, tau=0.4) on residuals AND raw returns
    # ------------------------------------------------------------------
    print("\n== Baseline (k=2.0, tau=0.4) ==")
    rows = []
    avalanche_dump = {}

    for label, series in [("residuals", residuals), ("raw_returns", log_returns)]:
        avalanches, summary, _ = run_one(
            series, label, DEFAULT_K, DEFAULT_TAU, args.rebuild_every
        )
        rows.append(summary)
        avalanche_dump[label] = avalanches
        print(
            f"  {label:>14s}: n={summary['n_avalanches']}, "
            f"sigma={summary['branching_ratio']:.3f} "
            f"[{summary['branching_ratio_lo']:.3f}, {summary['branching_ratio_hi']:.3f}], "
            f"alpha_size={summary.get('size_alpha', float('nan')):.3f}"
        )

        plot_ccdf(
            [a.size for a in avalanches],
            title=f"Avalanche size CCDF ({label}, k={DEFAULT_K}, τ={DEFAULT_TAU})",
            xlabel="avalanche size (stocks)",
            out_path=FIG_DIR / f"ccdf_size_{label}.png",
        )
        plot_ccdf(
            [a.duration for a in avalanches],
            title=f"Avalanche duration CCDF ({label}, k={DEFAULT_K}, τ={DEFAULT_TAU})",
            xlabel="avalanche duration (days)",
            out_path=FIG_DIR / f"ccdf_duration_{label}.png",
        )

        avalanches_to_frame(avalanches).to_csv(
            TAB_DIR / f"avalanches_{label}.csv", index=False
        )

    # ------------------------------------------------------------------
    # Robustness sweep (residuals only)
    # ------------------------------------------------------------------
    print("\n== Robustness sweep (residuals) ==")
    if args.quick:
        ks, taus = [1.5, 2.0, 2.5], [0.3, 0.4, 0.5]
    else:
        ks, taus = K_GRID, TAU_GRID

    sweep_rows = []
    for k in ks:
        for tau in taus:
            _, summary, _ = run_one(residuals, "residuals", k, tau, args.rebuild_every)
            sweep_rows.append(summary)
            print(
                f"  k={k:.1f}, tau={tau:.1f}: n={summary['n_avalanches']:>5d}, "
                f"sigma={summary['branching_ratio']:.3f}, "
                f"alpha_size={summary.get('size_alpha', float('nan')):.3f}"
            )

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(TAB_DIR / "robustness_sweep.csv", index=False)

    pd.DataFrame(rows).to_csv(TAB_DIR / "baseline_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Null model
    # ------------------------------------------------------------------
    if not args.no_null:
        print("\n== Null model (residuals) ==")
        null_cache = DATA_DIR / "null_shuffles.pkl"
        n_shuffles = 25 if args.quick else 100

        if args.null_from_cache and null_cache.exists():
            print(f"  loading cached null shuffles from {null_cache}")
            with open(null_cache, "rb") as f:
                null_out = pickle.load(f)
            n_shuffles = len(null_out["sizes_per_shuffle"])
        else:
            null_out = run_null_model(
                residuals,
                k=DEFAULT_K,
                tau=DEFAULT_TAU,
                window=ROLLING_WINDOW,
                n_shuffles=n_shuffles,
                seed=RANDOM_SEED,
                rebuild_every=args.rebuild_every,
            )
            # Persist BEFORE plotting/summarizing so a downstream bug
            # never costs us another 30+ minutes of shuffles.
            with open(null_cache, "wb") as f:
                pickle.dump(null_out, f)
            print(f"  cached null shuffles to {null_cache}")

        empirical_sizes = [a.size for a in avalanche_dump["residuals"]]
        plot_empirical_vs_null(
            empirical_sizes,
            null_out["sizes_per_shuffle"],
            title=f"Empirical vs null avalanche sizes (k={DEFAULT_K}, τ={DEFAULT_TAU})",
            out_path=FIG_DIR / "empirical_vs_null.png",
        )

        # Summary stats: empirical max vs null max distribution
        null_max = np.array([max(s) if s else 0 for s in null_out["sizes_per_shuffle"]])
        emp_max = max(empirical_sizes) if empirical_sizes else 0
        null_n = null_out["n_avalanches_per_shuffle"]
        with open(TAB_DIR / "null_summary.json", "w") as f:
            json.dump(
                {
                    "n_shuffles": n_shuffles,
                    "empirical_max_size": int(emp_max),
                    "empirical_n_avalanches": len(empirical_sizes),
                    "null_max_size_mean": float(null_max.mean()),
                    "null_max_size_p95": float(np.percentile(null_max, 95)),
                    "null_n_avalanches_mean": float(null_n.mean()),
                    "p_value_max_size": float((null_max >= emp_max).mean()),
                },
                f,
                indent=2,
            )
        print(f"  empirical max size: {emp_max}")
        print(f"  null max size: mean={null_max.mean():.1f}, p95={np.percentile(null_max,95):.1f}")

    print("\n== Done ==")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TAB_DIR}")


if __name__ == "__main__":
    main()
