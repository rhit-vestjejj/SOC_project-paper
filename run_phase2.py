"""
Phase 2 driver: rolling branching ratio + lead-lag + pre-crisis tests.

Reuses Phase 1 outputs:
    outputs/tables/avalanches_residuals.csv
    data/market_index.parquet

Usage:
    python run_phase2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, FIG_DIR, TAB_DIR
from src.plotting import plot_rolling_branching
from src.rolling_branching import (
    CRISIS_EVENTS,
    forward_realized_vol,
    incremental_predictive_regression,
    lead_lag_correlation,
    market_log_returns,
    precrisis_test,
    rolling_branching_ratio,
)

WINDOWS = [60, 120, 252]
HORIZONS = [21, 63, 126]
PRIMARY_WINDOW = 120


def main():
    print("== Loading Phase 1 outputs ==")
    av_path = TAB_DIR / "avalanches_residuals.csv"
    avalanches = pd.read_csv(av_path, parse_dates=["start_day", "end_day"])
    print(f"  {len(avalanches):,} avalanches from {av_path.name}")

    market = pd.read_parquet(DATA_DIR / "market_index.parquet")
    mkt_ret = market_log_returns(market)
    trading_days = mkt_ret.index
    print(f"  market series: {len(trading_days):,} days, "
          f"{trading_days.min().date()} → {trading_days.max().date()}")

    # ------------------------------------------------------------------
    # Rolling branching ratios
    # ------------------------------------------------------------------
    print("\n== Rolling branching ratio ==")
    sigma_by_w = {
        w: rolling_branching_ratio(avalanches, trading_days, window=w)
        for w in WINDOWS
    }
    for w, s in sigma_by_w.items():
        valid = s.dropna()
        print(f"  W = {w:>3d}d: n_valid={len(valid):>4d}  "
              f"min={valid.min():.3f}  median={valid.median():.3f}  "
              f"max={valid.max():.3f}")

    # Save the time series as a CSV
    sigma_df = pd.DataFrame(sigma_by_w)
    sigma_df.index.name = "date"
    sigma_df.to_csv(TAB_DIR / "rolling_branching_ratio.csv")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_rolling_branching(
        sigma_by_w,
        crisis_events=CRISIS_EVENTS,
        out_path=FIG_DIR / "rolling_branching_ratio.png",
        title="Rolling branching ratio σ(t) — residuals, k=2.0, τ=0.4",
        primary_window=PRIMARY_WINDOW,
    )
    print(f"\n  figure → {FIG_DIR/'rolling_branching_ratio.png'}")

    # ------------------------------------------------------------------
    # Lead-lag: sigma(t) vs forward realized volatility
    # ------------------------------------------------------------------
    print("\n== Lead-lag correlation: σ(t) vs forward realized vol ==")
    sigma_primary = sigma_by_w[PRIMARY_WINDOW]
    leadlag_rows = []
    for w in WINDOWS:
        for h in HORIZONS:
            fv = forward_realized_vol(mkt_ret, horizon=h)
            res = lead_lag_correlation(sigma_by_w[w], fv)
            leadlag_rows.append({
                "sigma_window": w,
                "vol_horizon": h,
                "n": res.n,
                "pearson_r": res.pearson_r,
                "pearson_p": res.pearson_p,
                "spearman_r": res.spearman_r,
                "spearman_p": res.spearman_p,
            })
    leadlag_df = pd.DataFrame(leadlag_rows)
    leadlag_df.to_csv(TAB_DIR / "lead_lag.csv", index=False)
    primary_view = leadlag_df[leadlag_df["sigma_window"] == PRIMARY_WINDOW]
    for _, r in primary_view.iterrows():
        print(f"  W=120  →  fwd vol [{int(r.vol_horizon):>3d}d]: "
              f"Pearson r={r.pearson_r:+.3f} (p={r.pearson_p:.2e}), "
              f"Spearman ρ={r.spearman_r:+.3f} (p={r.spearman_p:.2e}), "
              f"n={int(r.n)}")

    # ------------------------------------------------------------------
    # Incremental predictive regression: does σ retain predictive power
    # after controlling for trailing realized vol?
    # ------------------------------------------------------------------
    print("\n== Incremental predictive regression (HAC SEs) ==")
    print("    fwd_vol_h(t) = a + b·lagged_vol_h(t) + c·σ_120(t) + ε")
    inc_rows = []
    for h in HORIZONS:
        res = incremental_predictive_regression(sigma_primary, mkt_ret, horizon=h)
        inc_rows.append(res.__dict__)
        print(
            f"  h={h:>3d}d:  n={res.n:>4d}  "
            f"R² vol-only={res.r2_baseline:.3f}  "
            f"R² +σ={res.r2_full:.3f}  ΔR²={res.delta_r2:+.3f}  "
            f"c_σ={res.coef_sigma:+.4f} (t={res.t_sigma:+.2f}, "
            f"p_HAC={res.p_sigma:.2e})"
        )
    pd.DataFrame(inc_rows).to_csv(TAB_DIR / "incremental_predictive.csv", index=False)

    # ------------------------------------------------------------------
    # Pre-crisis test: is σ in the 60d before each crisis elevated?
    # ------------------------------------------------------------------
    print("\n== Pre-crisis Mann-Whitney (σ_120, lookback=60d, alt='greater') ==")
    pre_rows = []
    for label, date_str in CRISIS_EVENTS:
        res = precrisis_test(sigma_primary, pd.Timestamp(date_str), lookback=60)
        pre_rows.append({"event": label, "date": date_str, **res})
        print(f"  {label:<24s} {date_str}: "
              f"median_pre={res['median_pre']:.3f}  "
              f"median_rest={res['median_rest']:.3f}  "
              f"p={res['p_one_sided']:.4f}  "
              f"({'**' if res['p_one_sided']<0.05 else '  '})")
    pd.DataFrame(pre_rows).to_csv(TAB_DIR / "precrisis_test.csv", index=False)

    # ------------------------------------------------------------------
    # Summary blob
    # ------------------------------------------------------------------
    summary = {
        "windows": WINDOWS,
        "primary_window": PRIMARY_WINDOW,
        "horizons": HORIZONS,
        "n_avalanches": int(len(avalanches)),
        "sigma_overall_median": {w: float(sigma_by_w[w].median())
                                 for w in WINDOWS},
        "n_crisis_events_significant": int(
            sum(1 for r in pre_rows
                if r["p_one_sided"] is not None
                and not np.isnan(r["p_one_sided"])
                and r["p_one_sided"] < 0.05)
        ),
    }
    with open(TAB_DIR / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n== Done ==")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TAB_DIR}")


if __name__ == "__main__":
    main()
