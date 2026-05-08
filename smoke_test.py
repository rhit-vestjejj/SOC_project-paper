"""
End-to-end sanity check on synthetic data.

Generates random returns with a few injected synchronous shocks, runs the
full Phase 1 pipeline, and prints summary statistics.

Run:  python smoke_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.activation import detect_activations
from src.avalanche import detect_avalanches
from src.network import build_correlation_provider
from src.statistics import summarize


def synth_data(T: int = 800, N: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.01, size=(T, N))

    # Inject 30 synchronous shocks: a random subset of stocks all spike on the same day
    for _ in range(30):
        day = rng.integers(70, T)
        size = rng.integers(5, 20)
        cols = rng.choice(N, size=size, replace=False)
        base[day, cols] += rng.choice([-1, 1]) * rng.uniform(0.04, 0.08)
        # Also propagate to neighbors next day
        if day + 1 < T:
            n_prop = rng.integers(2, 8)
            cols2 = rng.choice(N, size=n_prop, replace=False)
            base[day + 1, cols2] += rng.choice([-1, 1]) * rng.uniform(0.03, 0.06)

    dates = pd.bdate_range("2010-01-01", periods=T)
    cols = [f"S{i:03d}" for i in range(N)]
    return pd.DataFrame(base, index=dates, columns=cols)


def main():
    print("Generating synthetic returns...")
    df = synth_data()
    print(f"Shape: {df.shape}")

    print("Detecting activations (k=2)...")
    activated = detect_activations(df, k=2.0, window=60)
    print(f"  total activations: {int(activated.values.sum())}")
    print(f"  active days: {(activated.sum(axis=1) > 0).sum()}")

    print("Building correlation network (tau=0.4)...")
    net = build_correlation_provider(df, tau=0.4, window=60, rebuild_every=5)

    print("Running avalanche detection...")
    avalanches = detect_avalanches(activated, net)
    print(f"  avalanches detected: {len(avalanches)}")

    if avalanches:
        sizes = [a.size for a in avalanches]
        durs = [a.duration for a in avalanches]
        print(f"  size: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.2f}")
        print(f"  duration: min={min(durs)}, max={max(durs)}, mean={np.mean(durs):.2f}")

    summary = summarize(avalanches)
    print("\nSummary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
