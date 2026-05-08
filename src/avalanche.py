"""
Phase 1.5: Avalanche detection.

BFS-style flood fill across consecutive trading days.

Definitions
-----------
- frontier_t(A) : stocks belonging to avalanche A that were activated on day t.
- A stock s activated on day t is "secondary" if there exists some live
  avalanche A with a frontier_{t-1}(A) member that is either s itself
  (continuation) or a network neighbor of s.
- Otherwise s is "primary" and seeds a new avalanche.
- If s connects to multiple live avalanches, those avalanches merge
  (union-find).
- An avalanche terminates when frontier_t becomes empty.

The detector is network-agnostic: it consumes any object with a
`neighbors_on(date) -> dict[ticker, set[ticker]]` method.

Same-day propagation (Phase 5)
------------------------------
With `same_day_propagation=True`, a stock activated on day t is also
counted as secondary if it is a network-neighbor of another stock
activated on the same day t. Within each connected component of the
same-day activation subgraph, exactly one canonical seed (the
sorted-min ticker) is primary; the other members are secondary. If
the component intersects yesterday's frontier of any live avalanche,
all members are secondary and the component is absorbed into that
avalanche. This addresses the daily-resolution bias that systematically
under-counts secondary activations when a cascade completes within one
trading day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Avalanche:
    id: int
    members: set = field(default_factory=set)
    activations: list = field(default_factory=list)  # list[(ticker, date, kind)]
    primary: int = 0
    secondary: int = 0
    start_day: pd.Timestamp | None = None
    end_day: pd.Timestamp | None = None

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def duration(self) -> int:
        if not self.activations:
            return 0
        days = {d for _, d, _ in self.activations}
        return len(days)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "size": self.size,
            "duration": self.duration,
            "primary": self.primary,
            "secondary": self.secondary,
            "start_day": self.start_day,
            "end_day": self.end_day,
            "n_activations": len(self.activations),
        }


def _make_unionfind(ids):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def _same_day_components(
    active_today: set[str], adj_map: dict[str, set[str]]
) -> list[set[str]]:
    """Connected components of `active_today` under the subgraph induced
    by `adj_map` restricted to active-today stocks. BFS over a dict-of-sets."""
    components: list[set[str]] = []
    seen: set[str] = set()
    for s in active_today:
        if s in seen:
            continue
        comp: set[str] = set()
        stack = [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            for n in adj_map.get(x, ()):
                if n in active_today and n not in seen:
                    stack.append(n)
        components.append(comp)
    return components


def detect_avalanches(
    activated: pd.DataFrame,
    network_provider,
    same_day_propagation: bool = False,
) -> list[Avalanche]:
    """
    Parameters
    ----------
    activated : DataFrame, 0/1, shape (T, N)
    network_provider : object with `.neighbors_on(date) -> dict`
    same_day_propagation : if True, treat network-connected same-day
        activations as part of the same avalanche component (one
        canonical seed primary, the rest secondary). See module
        docstring.

    Returns
    -------
    list[Avalanche]
    """
    avalanches: dict[int, Avalanche] = {}
    live: dict[int, set] = {}  # avalanche_id -> frontier yesterday
    next_id = 0

    cols = activated.columns.to_numpy()

    for date in activated.index:
        row = activated.loc[date].values.astype(bool)
        if not row.any():
            live = {}
            continue
        active_today = set(cols[row].tolist())

        adj_map = network_provider.neighbors_on(date)

        # Step 1: figure out which live avalanches each activated stock connects to
        connections: dict[str, set] = {}
        for s in active_today:
            nbrs = adj_map.get(s, set())
            conn = set()
            for av_id, frontier in live.items():
                if s in frontier or (nbrs & frontier):
                    conn.add(av_id)
            connections[s] = conn

        # Step 1b (Phase 5): same-day component augmentation.
        # Pre-allocated avalanche IDs for freestanding components
        # (component has no yesterday-frontier intersection); maps the
        # canonical seed ticker to the new avalanche id.
        fresh_seed_av_id: dict[str, int] = {}
        # Maps each non-seed member of a freestanding component to its seed.
        nonseed_to_seed: dict[str, str] = {}

        if same_day_propagation:
            components = _same_day_components(active_today, adj_map)
            for component in components:
                if len(component) < 2:
                    continue
                # Union of yesterday-connections across the component
                union_yesterday: set = set()
                for m in component:
                    union_yesterday |= connections[m]
                if union_yesterday:
                    # Component absorbed into yesterday avalanche(s).
                    # Sharing the union triggers Step-2 union-find merges
                    # exactly as if a single member had spanned them.
                    for m in component:
                        connections[m] = set(union_yesterday)
                else:
                    # Freestanding same-day component: deterministic seed.
                    seed = min(component)  # sorted ticker
                    target_id = next_id
                    next_id += 1
                    avalanches[target_id] = Avalanche(
                        id=target_id, start_day=date, end_day=date
                    )
                    fresh_seed_av_id[seed] = target_id
                    for m in component:
                        if m != seed:
                            nonseed_to_seed[m] = seed

        # Step 2: union-find merge of any avalanches that share a connecting stock
        parent, find, union = _make_unionfind(live.keys())
        for s, conn in connections.items():
            conn_list = list(conn)
            for j in range(1, len(conn_list)):
                union(conn_list[0], conn_list[j])

        # Apply merges to avalanche records
        if parent:
            roots_seen = set()
            for av_id in list(parent.keys()):
                root = find(av_id)
                if root == av_id:
                    roots_seen.add(root)
                    continue
                a_root = avalanches[root]
                a_other = avalanches[av_id]
                a_root.members |= a_other.members
                a_root.activations.extend(a_other.activations)
                a_root.primary += a_other.primary
                a_root.secondary += a_other.secondary
                if a_root.start_day is None or a_other.start_day < a_root.start_day:
                    a_root.start_day = a_other.start_day
                if a_root.end_day is None or a_other.end_day > a_root.end_day:
                    a_root.end_day = a_other.end_day
                del avalanches[av_id]

        # Step 3: assign each activation to its (root) avalanche, building today's frontier
        new_frontier: dict[int, set] = {}
        for s, conn in connections.items():
            if conn:
                root = find(next(iter(conn)))
                kind = "secondary"
                target_id = root
            elif s in fresh_seed_av_id:
                # Canonical seed of a freestanding same-day component:
                # primary, joins the pre-allocated avalanche.
                target_id = fresh_seed_av_id[s]
                kind = "primary"
            elif s in nonseed_to_seed:
                # Non-seed of a freestanding same-day component:
                # secondary, joins the seed's pre-allocated avalanche.
                target_id = fresh_seed_av_id[nonseed_to_seed[s]]
                kind = "secondary"
            else:
                # Singleton primary (no yesterday-conn, not in a
                # multi-member same-day component).
                target_id = next_id
                next_id += 1
                avalanches[target_id] = Avalanche(
                    id=target_id, start_day=date, end_day=date
                )
                kind = "primary"

            av = avalanches[target_id]
            av.members.add(s)
            av.activations.append((s, date, kind))
            av.end_day = date
            if kind == "primary":
                av.primary += 1
            else:
                av.secondary += 1
            new_frontier.setdefault(target_id, set()).add(s)

        live = new_frontier

    return list(avalanches.values())


def avalanches_to_frame(avalanches: list[Avalanche]) -> pd.DataFrame:
    if not avalanches:
        return pd.DataFrame(
            columns=[
                "id", "size", "duration", "primary", "secondary",
                "start_day", "end_day", "n_activations",
            ]
        )
    return pd.DataFrame([a.to_dict() for a in avalanches])
