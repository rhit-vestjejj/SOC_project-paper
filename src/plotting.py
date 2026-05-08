"""Plot helpers: log-log CCDFs with power-law fit overlay, null comparisons."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .statistics import ccdf, fit_power_law


def plot_size_duration_scaling(scaling_fits: list, out_path: Path,
                               title: str = "Size-duration scaling ⟨S | T⟩ ~ T^γ"):
    """Plot binned ⟨S|T⟩ vs T on log-log axes for one or more avalanche sets,
    each with its fitted line. `scaling_fits` is a list of ScalingFit objects."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["C0", "C3", "C2", "C1"]
    for i, sf in enumerate(scaling_fits):
        c = colors[i % len(colors)]
        if sf.binned.empty or np.isnan(sf.gamma):
            continue
        T = sf.binned["duration"].values.astype(float)
        S = np.exp(sf.binned["mean_log_size"].values)
        ax.loglog(T, S, "o", color=c, markersize=5, alpha=0.85,
                  label=f"{sf.label}: γ = {sf.gamma:.2f} ± {sf.gamma_se:.2f}  (n={sf.n_avalanches_used:,})")
        # Fitted line
        T_grid = np.geomspace(T.min(), T.max(), 50)
        S_fit = np.exp(sf.intercept) * T_grid ** sf.gamma
        ax.loglog(T_grid, S_fit, "-", color=c, linewidth=1.2, alpha=0.6)
    ax.set_xlabel("avalanche duration T (trading days)")
    ax.set_ylabel("⟨ avalanche size | T ⟩")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kt_heatmap(sweep_df: pd.DataFrame, out_path: Path,
                    baseline_k: float = 2.0, baseline_tau: float = 0.4):
    """Two-panel heatmap of σ and α_size over the (k, τ) grid.

    Reads `sweep_df` produced by run_baseline.py / run_phase5.py:
    requires columns 'k', 'tau', 'branching_ratio', 'size_alpha'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, col, title, fmt in [
        (axes[0], "branching_ratio", "Branching ratio σ", ".2f"),
        (axes[1], "size_alpha", "Power-law exponent α (size)", ".2f"),
    ]:
        pivot = sweep_df.pivot(index="k", columns="tau", values=col).sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])
        ax.set_xlabel("τ")
        ax.set_ylabel("k")
        ax.set_title(title)
        for i, k in enumerate(pivot.index):
            for j, t in enumerate(pivot.columns):
                v = pivot.iloc[i, j]
                if pd.isna(v):
                    continue
                txt_color = "white" if (v - np.nanmin(pivot.values)) / (np.nanmax(pivot.values) - np.nanmin(pivot.values) + 1e-9) < 0.55 else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color=txt_color, fontsize=9)
                if abs(k - baseline_k) < 1e-6 and abs(t - baseline_tau) < 1e-6:
                    ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                                fill=False, edgecolor="red", linewidth=2))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("(k, τ) parameter robustness — red box marks baseline (2.0, 0.4)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ccdf(
    sizes,
    title: str,
    xlabel: str,
    out_path: Path,
    fit: bool = True,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    x, p = ccdf(sizes)
    if len(x) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    ax.loglog(x, p, "o", markersize=4, label="empirical", alpha=0.8)

    if fit:
        f = fit_power_law(sizes)
        if f is not None and f.alpha > 1:
            xs = np.logspace(np.log10(f.xmin), np.log10(x.max()), 100)
            tail_frac = f.n_tail / f.n
            ys = tail_frac * (xs / f.xmin) ** (-(f.alpha - 1))
            ax.loglog(
                xs,
                ys,
                "r--",
                label=f"power law (α={f.alpha:.2f}, x_min={f.xmin:.0f})",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ccdf_comparison(
    series_dict: dict,
    title: str,
    xlabel: str,
    out_path: Path,
    axis_mode: str = "loglog",
    xlim: tuple | None = None,
):
    """
    series_dict : dict[label, list_of_values]   each label gets its own CCDF
    axis_mode   : "loglog" (default), "semilog" (linear x, log y), or
                  "linear" (both linear). Use semilog when you want the
                  natural x-axis but still need to see the heavy tail.
    xlim        : optional (xmin, xmax) for x-axis
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["C0", "C3", "C2", "C1"]
    if axis_mode == "loglog":
        plot_fn = ax.loglog
    elif axis_mode == "semilog":
        plot_fn = ax.semilogy
    elif axis_mode == "linear":
        plot_fn = ax.plot
    else:
        raise ValueError(f"axis_mode must be loglog/semilog/linear, got {axis_mode!r}")
    for i, (label, values) in enumerate(series_dict.items()):
        x, p = ccdf(values)
        if not len(x):
            continue
        plot_fn(x, p, "o-", color=colors[i % len(colors)], markersize=4,
                linewidth=1.2, alpha=0.85, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if axis_mode == "linear":
        ax.set_ylim(0, 1.02)
    ax.grid(True, which="both" if axis_mode != "linear" else "major", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_branching(
    series_by_window: dict,
    crisis_events: list,
    out_path: Path,
    title: str = "Rolling branching ratio",
    primary_window: int | None = None,
):
    """
    series_by_window : dict[int, pd.Series]   window length -> sigma_W(t)
    crisis_events    : list[(label, date_str)]
    primary_window   : if given, that window is drawn bold; others faint.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for w, s in series_by_window.items():
        if primary_window is not None and w != primary_window:
            ax.plot(s.index, s.values, alpha=0.45, linewidth=0.9, label=f"W = {w}d")
        else:
            ax.plot(s.index, s.values, linewidth=1.6, label=f"W = {w}d")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.7,
               label="σ = 1 (critical)")

    # Crisis vlines + labels
    ymin, ymax = ax.get_ylim()
    for label, date_str in crisis_events:
        d = pd.Timestamp(date_str)
        ax.axvline(d, color="C3", linewidth=0.7, alpha=0.55)
        ax.text(d, ymax * 0.97, label, rotation=90, fontsize=8,
                color="C3", va="top", ha="right", alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("branching ratio σ")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_empirical_vs_null(
    empirical_sizes,
    null_sizes_per_shuffle,
    title: str,
    out_path: Path,
    axis_mode: str = "loglog",
    xlim: tuple | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    if axis_mode == "loglog":
        plot_fn = ax.loglog
    elif axis_mode == "semilog":
        plot_fn = ax.semilogy
    elif axis_mode == "linear":
        plot_fn = ax.plot
    else:
        raise ValueError(f"axis_mode must be loglog/semilog/linear, got {axis_mode!r}")

    null_xs = []
    null_ps = []
    for sizes in null_sizes_per_shuffle:
        x, p = ccdf(sizes)
        if len(x):
            null_xs.append(x)
            null_ps.append(p)

    if null_xs:
        for x, p in zip(null_xs[:30], null_ps[:30]):
            plot_fn(x, p, color="grey", alpha=0.15, linewidth=0.7)
        plot_fn([], [], color="grey", alpha=0.5, label="null shuffles")

    x, p = ccdf(empirical_sizes)
    if len(x):
        plot_fn(x, p, "o-", color="C3", markersize=4, label="empirical", linewidth=1.5)

    ax.set_xlabel("avalanche size")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if axis_mode == "linear":
        ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, which="both" if axis_mode != "linear" else "major", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
