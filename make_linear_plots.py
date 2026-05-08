"""
Re-emit the four comparison plots with linear x/y axes (no log-log).
Useful when the log-log "rakes the tail flat" presentation distorts
the visceral message about how far empirical avalanches reach beyond
the various null/baseline distributions.

Reuses pickled and CSV outputs from Phases 1, 3, 4, 5 — does not rerun
detection.

Usage:
    python make_linear_plots.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, FIG_DIR, TAB_DIR
from src.plotting import plot_ccdf_comparison, plot_empirical_vs_null

XMAX = 500


def main():
    # ------------------------------------------------------------------
    # Phase 1 baseline: empirical vs 100-shuffle null (log-log version
    # already exists at empirical_vs_null.png)
    # ------------------------------------------------------------------
    p1_av = pd.read_csv(TAB_DIR / "avalanches_residuals.csv")
    p1_sizes = p1_av["size"].tolist()

    # Phase 5 same-day: empirical vs 100-shuffle null (Phase 1 version
    # is superseded by this — we only keep the same-day null going
    # forward in the figure set)
    null_sd_pkl = DATA_DIR / "null_shuffles_sameday.pkl"
    if null_sd_pkl.exists():
        with open(null_sd_pkl, "rb") as f:
            null_sd = pickle.load(f)
        sd_av = pd.read_csv(TAB_DIR / "avalanches_residuals_sameday.csv")
        plot_empirical_vs_null(
            sd_av["size"].tolist(),
            null_sd["sizes_per_shuffle"],
            title="Empirical vs null avalanche sizes — Phase 5 same-day (natural x)",
            out_path=FIG_DIR / "empirical_vs_null_sameday_natx.png",
            axis_mode="semilog",
            xlim=(0, XMAX),
        )
        print(f"  → empirical_vs_null_sameday_natx.png")

    # ------------------------------------------------------------------
    # Phase 3 comparison: raw vs partial correlation
    # ------------------------------------------------------------------
    p3_paths = {
        f"raw corr τ=0.4 (Phase 1)": p1_sizes,
    }
    for tau in [0.05, 0.03]:
        p = TAB_DIR / f"avalanches_partial_tau{tau:.2f}.csv"
        if p.exists():
            p3_paths[f"partial corr τ={tau}"] = pd.read_csv(p)["size"].tolist()
    if len(p3_paths) > 1:
        plot_ccdf_comparison(
            p3_paths,
            title="Phase 3: raw vs partial correlation (natural x)",
            xlabel="avalanche size (stocks)",
            out_path=FIG_DIR / "ccdf_comparison_phase3_natx.png",
            axis_mode="semilog",
            xlim=(0, XMAX),
        )
        print(f"  → ccdf_comparison_phase3_natx.png")

    # ------------------------------------------------------------------
    # Phase 4 comparison: empirical vs Hawkes-simulated
    # ------------------------------------------------------------------
    hawkes_pkl = DATA_DIR / "hawkes_sim_sizes.pkl"
    if hawkes_pkl.exists():
        with open(hawkes_pkl, "rb") as f:
            hk = pickle.load(f)
        pooled_sim = [s for sizes in hk["sizes_per_sim"] for s in sizes]
        plot_ccdf_comparison(
            {
                "empirical (Phase 1)": p1_sizes,
                "Hawkes-simulated (pooled)": pooled_sim,
            },
            title="Phase 4: empirical vs Hawkes-simulated (natural x)",
            xlabel="avalanche size (stocks)",
            out_path=FIG_DIR / "ccdf_empirical_vs_hawkes_natx.png",
            axis_mode="semilog",
            xlim=(0, XMAX),
        )
        print(f"  → ccdf_empirical_vs_hawkes_natx.png")

    # ------------------------------------------------------------------
    # Phase 1 vs Phase 5 same-day baseline (a fresh comparison plot)
    # ------------------------------------------------------------------
    sd_csv = TAB_DIR / "avalanches_residuals_sameday.csv"
    if sd_csv.exists():
        sd_sizes = pd.read_csv(sd_csv)["size"].tolist()
        plot_ccdf_comparison(
            {
                "Phase 1 baseline (no same-day)": p1_sizes,
                "Phase 5 same-day propagation": sd_sizes,
            },
            title="Phase 1 vs Phase 5: detector-rule effect (natural x)",
            xlabel="avalanche size (stocks)",
            out_path=FIG_DIR / "ccdf_phase1_vs_phase5_natx.png",
            axis_mode="semilog",
            xlim=(0, XMAX),
        )
        print(f"  → ccdf_phase1_vs_phase5_natx.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
