"""
Phase 3 driver: rerun avalanche pipeline on a partial-correlation
network (LedoitWolf shrinkage) and compare to the Phase 1 raw-corr
baseline.

The key question (per agents.md §3): do power-law avalanches survive
the network upgrade? If yes, the SOC story is robust to common-factor
deconfounding. If the heavy tail collapses, Phase 1 was driven by
spurious indirect edges.

Reuses cached residuals from data/residuals.parquet.

Usage:
    python run_phase3.py
    python run_phase3.py --tau 0.05 --tau-extra 0.03
"""
from __future__ import annotations

import argparse
import json
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
    RANDOM_SEED,
    ROLLING_WINDOW,
    TAB_DIR,
)
from src.partial_corr import build_partial_corr_provider
from src.plotting import plot_ccdf, plot_ccdf_comparison
from src.statistics import summarize


def run_partial_corr(
    residuals: pd.DataFrame,
    tau_partial: float,
    rebuild_every: int,
):
    activated = detect_activations(residuals, k=DEFAULT_K, window=ROLLING_WINDOW)
    net = build_partial_corr_provider(
        residuals, tau=tau_partial, window=ROLLING_WINDOW, rebuild_every=rebuild_every
    )
    avalanches = detect_avalanches(activated, net)

    # Sample density across rebuild dates
    sample_dates = residuals.index[ROLLING_WINDOW :: max(1, len(residuals) // 30)]
    densities = []
    for d in sample_dates:
        net.neighbors_on(d)
        if not np.isnan(net.last_density):
            densities.append(net.last_density)
    median_density = float(np.median(densities)) if densities else float("nan")

    summary = summarize(avalanches)
    summary.update({
        "label": f"partial_corr_tau{tau_partial:.2f}",
        "k": DEFAULT_K,
        "tau_partial": tau_partial,
        "median_edge_density": median_density,
    })
    return avalanches, summary


def load_baseline_avalanche_sizes() -> list[int]:
    df = pd.read_csv(TAB_DIR / "avalanches_residuals.csv")
    return df["size"].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.05,
                        help="primary partial-corr threshold (matched density)")
    parser.add_argument("--tau-extra", type=float, default=0.03,
                        help="secondary threshold for robustness")
    parser.add_argument("--rebuild-every", type=int, default=5)
    parser.add_argument("--single-tau", action="store_true",
                        help="run only the primary tau (faster)")
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    print("== Loading residuals ==")
    residuals = pd.read_parquet(DATA_DIR / "residuals.parquet")
    print(f"  shape: {residuals.shape}")

    taus = [args.tau] if args.single_tau else [args.tau, args.tau_extra]

    print(f"\n== Running partial-corr pipeline at τ ∈ {taus} ==")
    summaries = []
    avalanche_sets = {}
    for tau in taus:
        print(f"\n  --- τ_partial = {tau} ---")
        avs, summary = run_partial_corr(residuals, tau, args.rebuild_every)
        summaries.append(summary)
        avalanche_sets[tau] = avs
        print(
            f"    n_avalanches = {summary['n_avalanches']:,}  "
            f"median_edge_density = {summary['median_edge_density']:.4f}  "
            f"σ = {summary['branching_ratio']:.3f}  "
            f"α_size = {summary.get('size_alpha', float('nan')):.3f}  "
            f"size_max = {summary['size_max']}"
        )

        avalanches_to_frame(avs).to_csv(
            TAB_DIR / f"avalanches_partial_tau{tau:.2f}.csv", index=False
        )
        plot_ccdf(
            [a.size for a in avs],
            title=f"Phase 3 avalanche size CCDF (partial corr, τ={tau})",
            xlabel="avalanche size (stocks)",
            out_path=FIG_DIR / f"ccdf_size_partial_tau{tau:.2f}.png",
        )

    # ------------------------------------------------------------------
    # Save summary + comparison plot
    # ------------------------------------------------------------------
    pd.DataFrame(summaries).to_csv(TAB_DIR / "phase3_summary.csv", index=False)

    # Comparison CCDF: baseline vs partial corr (one or two τ's)
    baseline_sizes = load_baseline_avalanche_sizes()
    series_dict = {
        f"raw corr τ={DEFAULT_TAU} (Phase 1)": baseline_sizes,
    }
    for tau in taus:
        series_dict[f"partial corr τ={tau}"] = [a.size for a in avalanche_sets[tau]]
    plot_ccdf_comparison(
        series_dict,
        title="Avalanche size CCDF: raw vs partial correlation",
        xlabel="avalanche size (stocks)",
        out_path=FIG_DIR / "ccdf_comparison_phase3.png",
    )

    # JSON summary blob
    with open(TAB_DIR / "phase3_summary.json", "w") as f:
        json.dump(
            {
                "rebuild_every": args.rebuild_every,
                "summaries": [
                    {k: (v if not isinstance(v, np.generic) else v.item())
                     for k, v in s.items()}
                    for s in summaries
                ],
                "baseline_n_avalanches": len(baseline_sizes),
                "baseline_size_max": int(max(baseline_sizes)) if baseline_sizes else 0,
            },
            f,
            indent=2,
            default=str,
        )

    print("\n== Done ==")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TAB_DIR}")


if __name__ == "__main__":
    main()
