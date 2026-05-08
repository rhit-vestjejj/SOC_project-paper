"""
Phase 3: Partial-correlation network via LedoitWolf shrinkage.

Same interface as `NetworkProvider` (it exposes `neighbors_on(date)` and
`neighbors(date, ticker)`), so the avalanche detector consumes either
implementation interchangeably.

We use LedoitWolf rather than GraphicalLassoCV (per agents.md §3.1
option B): GraphicalLassoCV with 500 tickers × 60-day windows is slow
and unstable, while LedoitWolf is always invertible and fast enough for
~1,000 rolling rebuilds.

Drop-any-NaN policy
-------------------
LedoitWolf needs a complete numeric matrix, so for each rebuild window
we drop any stock with a missing value in that window. This is slightly
stricter than the pairwise-pandas-corr policy used in Phase 1 — it
shrinks the per-window universe a little but produces a cleaner
precision matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import ROLLING_WINDOW

DEFAULT_TAU_PARTIAL = 0.05  # calibrated so partial-corr edge density
                            # (~0.3 %) is comparable to the Phase 1 baseline
                            # raw-corr density at τ_raw = 0.4.


def precision_to_partial_corr(precision: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(precision))
    pc = -precision / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return pc


@dataclass
class PartialCorrProvider:
    series: pd.DataFrame
    tau: float = DEFAULT_TAU_PARTIAL
    window: int = ROLLING_WINDOW
    rebuild_every: int = 5
    min_obs_required: int = 30  # min rows in window after dropping cols

    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _last_rebuild_idx: int = field(default=-10**9, init=False, repr=False)
    _last_neighbors: dict = field(default_factory=dict, init=False, repr=False)
    _last_density: float = field(default=float("nan"), init=False, repr=False)

    def _date_idx(self, date) -> int:
        return self.series.index.get_loc(date)

    def _build(self, idx: int) -> dict[str, set[str]]:
        start = max(0, idx - self.window + 1)
        win = self.series.iloc[start : idx + 1]
        if len(win) < self.min_obs_required:
            self._last_density = float("nan")
            return {}

        # Drop any stock with any NaN in this window
        valid = win.dropna(axis=1, how="any")
        if valid.shape[1] < 2 or valid.shape[0] < self.min_obs_required:
            self._last_density = float("nan")
            return {}

        try:
            lw = LedoitWolf().fit(valid.values)
        except Exception:
            self._last_density = float("nan")
            return {}

        pc = precision_to_partial_corr(lw.precision_)
        adj = np.abs(pc) > self.tau

        cols = valid.columns.tolist()
        n = len(cols)
        n_edges = int(np.triu(adj, k=1).sum())
        possible = n * (n - 1) // 2
        self._last_density = n_edges / possible if possible else float("nan")

        neighbors: dict[str, set[str]] = {}
        for i, t in enumerate(cols):
            idxs = np.where(adj[i])[0]
            if len(idxs):
                neighbors[t] = {cols[j] for j in idxs}
        return neighbors

    def neighbors_on(self, date) -> dict[str, set[str]]:
        idx = self._date_idx(date)
        if idx - self._last_rebuild_idx >= self.rebuild_every or not self._last_neighbors:
            self._last_neighbors = self._build(idx)
            self._last_rebuild_idx = idx
        return self._last_neighbors

    def neighbors(self, date, ticker: str) -> set[str]:
        return self.neighbors_on(date).get(ticker, set())

    @property
    def last_density(self) -> float:
        return self._last_density

    def density_series(self, dates: Iterable | None = None) -> pd.Series:
        dates = list(dates) if dates is not None else self.series.index[self.window :]
        out: dict = {}
        for d in dates:
            self.neighbors_on(d)
            out[d] = self._last_density
        return pd.Series(out, name="edge_density")


def build_partial_corr_provider(
    series: pd.DataFrame,
    tau: float = DEFAULT_TAU_PARTIAL,
    window: int = ROLLING_WINDOW,
    rebuild_every: int = 5,
) -> PartialCorrProvider:
    return PartialCorrProvider(
        series=series, tau=tau, window=window, rebuild_every=rebuild_every
    )
