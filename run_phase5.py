"""
Phase 5 driver:

1. Same-day-propagation baseline + 16-cell (k, τ) sweep on residuals.
2. 100-shuffle null model with same-day propagation.
3. Crackling-noise size-duration scaling on three avalanche sets:
   Phase 1 (existing CSV), Phase 5 same-day, Phase 3 partial-corr.
4. (k, τ) heatmap of σ and α from the same-day sweep.

Reuses Phase 1 caches (data/residuals.parquet, etc.).

Usage:
    python run_phase5.py
    python run_phase5.py --no-null         # skip the 30-min step
    python run_phase5.py --null-from-cache # reuse data/null_shuffles_sameday.pkl
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
from src.network import build_correlation_provider
from src.null_model import run_null_model
from src.plotting import (
    plot_ccdf,
    plot_empirical_vs_null,
    plot_kt_heatmap,
    plot_size_duration_scaling,
)
from src.scaling import size_duration_scaling
from src.statistics import summarize


def run_one(
    series: pd.DataFrame,
    label: str,
    k: float,
    tau: float,
    rebuild_every: int,
    same_day: bool,
):
    activated = detect_activations(series, k=k, window=ROLLING_WINDOW)
    net = build_correlation_provider(
        series, tau=tau, window=ROLLING_WINDOW, rebuild_every=rebuild_every
    )
    avalanches = detect_avalanches(activated, net, same_day_propagation=same_day)
    summary = summarize(avalanches)
    summary.update({"label": label, "k": k, "tau": tau})
    return avalanches, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-null", action="store_true")
    parser.add_argument("--null-from-cache", action="store_true")
    parser.add_argument("--rebuild-every", type=int, default=5)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    print("== Loading residuals ==")
    residuals = pd.read_parquet(DATA_DIR / "residuals.parquet")
    print(f"  shape: {residuals.shape}")

    # ------------------------------------------------------------------
    # Same-day baseline (k=2, τ=0.4)
    # ------------------------------------------------------------------
    print("\n== Same-day baseline (k=2.0, τ=0.4) ==")
    avalanches_sd, summary_sd = run_one(
        residuals, "residuals_sameday",
        DEFAULT_K, DEFAULT_TAU, args.rebuild_every, same_day=True,
    )
    print(
        f"  n={summary_sd['n_avalanches']:,}  "
        f"σ = {summary_sd['branching_ratio']:.3f} "
        f"[{summary_sd['branching_ratio_lo']:.3f}, "
        f"{summary_sd['branching_ratio_hi']:.3f}]  "
        f"α = {summary_sd.get('size_alpha', float('nan')):.3f}  "
        f"max_size = {summary_sd['size_max']}"
    )

    avalanche_df_sd = avalanches_to_frame(avalanches_sd)
    avalanche_df_sd.to_csv(TAB_DIR / "avalanches_residuals_sameday.csv", index=False)

    pd.DataFrame([summary_sd]).to_csv(
        TAB_DIR / "baseline_summary_sameday.csv", index=False
    )

    plot_ccdf(
        [a.size for a in avalanches_sd],
        title=f"Avalanche size CCDF (residuals, same-day, k={DEFAULT_K}, τ={DEFAULT_TAU})",
        xlabel="avalanche size (stocks)",
        out_path=FIG_DIR / "ccdf_size_residuals_sameday.png",
    )

    # ------------------------------------------------------------------
    # Same-day robustness sweep (residuals)
    # ------------------------------------------------------------------
    print("\n== Same-day robustness sweep (residuals) ==")
    sweep_rows = []
    for k in K_GRID:
        for tau in TAU_GRID:
            _, summary = run_one(
                residuals, "residuals_sameday", k, tau, args.rebuild_every, same_day=True
            )
            sweep_rows.append(summary)
            print(
                f"  k={k:.1f}  τ={tau:.1f}: n={summary['n_avalanches']:>5d}  "
                f"σ={summary['branching_ratio']:.3f}  "
                f"α={summary.get('size_alpha', float('nan')):.3f}"
            )
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(TAB_DIR / "robustness_sweep_sameday.csv", index=False)

    # ------------------------------------------------------------------
    # (k, τ) heatmap (uses the same-day sweep)
    # ------------------------------------------------------------------
    plot_kt_heatmap(
        sweep_df,
        out_path=FIG_DIR / "kt_heatmap.png",
        baseline_k=DEFAULT_K,
        baseline_tau=DEFAULT_TAU,
    )
    print(f"\n  heatmap → {FIG_DIR / 'kt_heatmap.png'}")

    # ------------------------------------------------------------------
    # Same-day null model
    # ------------------------------------------------------------------
    if not args.no_null:
        null_cache = DATA_DIR / "null_shuffles_sameday.pkl"
        if args.null_from_cache and null_cache.exists():
            print(f"\n== Loading cached same-day null shuffles ==")
            with open(null_cache, "rb") as f:
                null_out = pickle.load(f)
            n_shuffles = len(null_out["sizes_per_shuffle"])
        else:
            print("\n== Same-day null model (100 shuffles) ==")
            n_shuffles = 100
            null_out = run_null_model(
                residuals,
                k=DEFAULT_K,
                tau=DEFAULT_TAU,
                window=ROLLING_WINDOW,
                n_shuffles=n_shuffles,
                seed=RANDOM_SEED,
                rebuild_every=args.rebuild_every,
                same_day_propagation=True,
            )
            with open(null_cache, "wb") as f:
                pickle.dump(null_out, f)
            print(f"  cached → {null_cache}")

        empirical_sizes_sd = [a.size for a in avalanches_sd]
        plot_empirical_vs_null(
            empirical_sizes_sd,
            null_out["sizes_per_shuffle"],
            title=f"Empirical vs null avalanche sizes — same-day (k={DEFAULT_K}, τ={DEFAULT_TAU})",
            out_path=FIG_DIR / "empirical_vs_null_sameday.png",
        )

        null_max = np.array([max(s) if s else 0 for s in null_out["sizes_per_shuffle"]])
        emp_max = max(empirical_sizes_sd) if empirical_sizes_sd else 0
        null_n = null_out["n_avalanches_per_shuffle"]
        null_summary = {
            "n_shuffles": int(n_shuffles),
            "empirical_max_size": int(emp_max),
            "empirical_n_avalanches": int(len(empirical_sizes_sd)),
            "null_max_size_mean": float(null_max.mean()),
            "null_max_size_p95": float(np.percentile(null_max, 95)),
            "null_n_avalanches_mean": float(null_n.mean()),
            "p_value_max_size": float((null_max >= emp_max).mean()),
        }
        with open(TAB_DIR / "null_summary_sameday.json", "w") as f:
            json.dump(null_summary, f, indent=2)
        print(
            f"  empirical max: {emp_max}  "
            f"null max: mean={null_max.mean():.1f}, p95={np.percentile(null_max,95):.1f}  "
            f"p_value={null_summary['p_value_max_size']:.4f}"
        )

    # ------------------------------------------------------------------
    # Crackling-noise size-duration scaling
    # ------------------------------------------------------------------
    print("\n== Size-duration scaling ⟨S | T⟩ ~ T^γ ==")
    scaling_fits = []

    # Phase 1 baseline avalanches (existing csv)
    p1_path = TAB_DIR / "avalanches_residuals.csv"
    if p1_path.exists():
        df = pd.read_csv(p1_path)
        sf = size_duration_scaling(df, label="Phase 1 baseline (no same-day)")
        scaling_fits.append(sf)
        print(
            f"  Phase 1 baseline:        γ = {sf.gamma:.3f} ± {sf.gamma_se:.3f}  "
            f"(n_bins={sf.n_bins}, n_used={sf.n_avalanches_used:,})"
        )

    # Phase 5 same-day avalanches
    sf = size_duration_scaling(avalanche_df_sd, label="Phase 5 same-day propagation")
    scaling_fits.append(sf)
    print(
        f"  Phase 5 same-day:        γ = {sf.gamma:.3f} ± {sf.gamma_se:.3f}  "
        f"(n_bins={sf.n_bins}, n_used={sf.n_avalanches_used:,})"
    )

    # Phase 3 partial-correlation avalanches at τ=0.03 (existing csv)
    p3_path = TAB_DIR / "avalanches_partial_tau0.03.csv"
    if p3_path.exists():
        df = pd.read_csv(p3_path)
        sf = size_duration_scaling(df, label="Phase 3 partial-corr τ=0.03")
        scaling_fits.append(sf)
        print(
            f"  Phase 3 partial-corr:    γ = {sf.gamma:.3f} ± {sf.gamma_se:.3f}  "
            f"(n_bins={sf.n_bins}, n_used={sf.n_avalanches_used:,})"
        )

    plot_size_duration_scaling(
        scaling_fits,
        out_path=FIG_DIR / "size_duration_scaling.png",
    )
    pd.DataFrame([
        {
            "label": sf.label,
            "gamma": sf.gamma,
            "gamma_se": sf.gamma_se,
            "intercept": sf.intercept,
            "n_bins": sf.n_bins,
            "n_avalanches_used": sf.n_avalanches_used,
            "n_avalanches_total": sf.n_avalanches_total,
        }
        for sf in scaling_fits
    ]).to_csv(TAB_DIR / "scaling_exponents.csv", index=False)

    print("\n== Done ==")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TAB_DIR}")


if __name__ == "__main__":
    main()
