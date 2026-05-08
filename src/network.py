"""
Phase 1.4: Rolling correlation network.

Designed to be modular: the avalanche detector consumes any object that
implements `neighbors(date, ticker) -> set[str]`. Different implementations
(rolling correlation, partial correlation in Phase 3) plug in interchangeably.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .config import DEFAULT_TAU, ROLLING_WINDOW


@dataclass
class NetworkProvider:
    """
    Lazy per-day adjacency. Recomputes the correlation matrix every
    `rebuild_every` trading days and reuses it in between.

    Stocks with insufficient data in a window are dropped from that window's
    adjacency.
    """

    series: pd.DataFrame  # rows = trading days, cols = tickers
    tau: float = DEFAULT_TAU
    window: int = ROLLING_WINDOW
    rebuild_every: int = 1
    min_periods: int = 30

    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _last_rebuild_idx: int = field(default=-10**9, init=False, repr=False)
    _last_neighbors: dict = field(default_factory=dict, init=False, repr=False)

    def _date_idx(self, date) -> int:
        return self.series.index.get_loc(date)

    def _build(self, idx: int) -> dict[str, set[str]]:
        start = max(0, idx - self.window + 1)
        win = self.series.iloc[start : idx + 1]
        # Need enough rows
        if len(win) < self.min_periods:
            return {}
        valid = win.dropna(axis=1, thresh=self.min_periods)
        if valid.shape[1] < 2:
            return {}
        corr = np.asarray(valid.corr().values, dtype=float).copy()
        np.fill_diagonal(corr, 0.0)
        adj = np.abs(corr) > self.tau
        cols = valid.columns.tolist()
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

    # ------------------------------------------------------------------
    # Bulk builder for diagnostics (memory hungry — use with care)
    # ------------------------------------------------------------------
    def density_series(self, dates: Iterable | None = None) -> pd.Series:
        """Fraction of nonzero edges per rebuild date."""
        dates = list(dates) if dates is not None else self.series.index[self.window :]
        out: dict = {}
        for d in dates:
            adj_map = self.neighbors_on(d)
            n_nodes = max(len(adj_map), 1)
            n_edges = sum(len(v) for v in adj_map.values()) // 2
            possible = n_nodes * (n_nodes - 1) / 2
            out[d] = n_edges / possible if possible else np.nan
        return pd.Series(out, name="edge_density")


@dataclass
class CachedNetwork:
    """
    Pre-computed adjacency-by-date snapshot. Built once from any
    NetworkProvider; thereafter each `neighbors_on(date)` is an O(1)
    dict lookup. Use this when running many independent simulations
    against the same empirical network (Phase 4).
    """
    snapshots: dict  # date -> dict[ticker, set[ticker]]

    def neighbors_on(self, date) -> dict[str, set[str]]:
        return self.snapshots.get(date, {})

    def neighbors(self, date, ticker: str) -> set[str]:
        return self.snapshots.get(date, {}).get(ticker, set())


def build_cached_network(provider, dates) -> CachedNetwork:
    snapshots: dict = {}
    last_signature: int | None = None
    last_adj: dict = {}
    for d in dates:
        adj = provider.neighbors_on(d)
        # neighbors_on returns the same dict object across consecutive
        # within-rebuild dates, so we share the reference (memory-cheap).
        sig = id(adj)
        if sig != last_signature:
            last_adj = {k: set(v) for k, v in adj.items()}  # defensive copy
            last_signature = sig
        snapshots[d] = last_adj
    return CachedNetwork(snapshots=snapshots)


def build_correlation_provider(
    series: pd.DataFrame,
    tau: float = DEFAULT_TAU,
    window: int = ROLLING_WINDOW,
    rebuild_every: int = 1,
) -> NetworkProvider:
    return NetworkProvider(
        series=series, tau=tau, window=window, rebuild_every=rebuild_every
    )
