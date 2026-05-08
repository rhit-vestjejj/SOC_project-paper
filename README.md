# Volatility Avalanches in S&P 500 Stock Networks

Are large stock-market drawdowns just bigger versions of small drawdowns,
or is there a moment of *coordinated cascade* — where a shock to one
stock spreads through a network of correlated peers and produces a fat
tail of "avalanches" much larger than any individual move?

That's the question this project tries to answer empirically, on
twenty-one years of daily S&P 500 data (2005–2025). The framework comes
from physics: **self-organized criticality** (SOC). The headline
finding is that yes — once you account for common factors, the
avalanches are real, network-mediated, and the system sits very close
to the SOC critical line.

This repository is the full pipeline: data ingest, network construction,
avalanche detection, statistics, four independent null/comparison
models, and a final robustness phase that addresses the two main
remaining weaknesses.

---

## The headline result, in one paragraph

After residualising returns against the market index, the stock's GICS
sector, and the Fama-French SMB/HML factors, we detect avalanches by
flooding outlier activations across a rolling correlation network.
Sizes follow a heavy-tailed distribution out to 485 stocks (≈ the
entire universe). The branching ratio σ — the SOC order parameter —
sits at **0.95 [0.80, 1.14]**, with the critical line σ = 1 inside the
confidence interval. The size–duration scaling exponent γ ≈ 1.7–2.0 is
stable across three independent avalanche-detection variants — exactly
the universality signature SOC predicts and that a lognormal alternative
cannot mimic. Empirical avalanches dominate three independent null
models (per-stock time shuffle, partial-correlation deconfounding,
Hawkes-Poisson identity). They do *not*, however, function as
forward-looking crash predictors once you control for trailing
volatility — σ is a contemporaneous regime gauge, not an early-warning
trigger. Both the affirmative and the negative findings are reported.

---

## Quick start

```bash
# One-time setup
pip install -r requirements.txt

# Phase 1 — full pipeline (data download + 100-shuffle null model)
# This is the long one: ~30–60 minutes, mostly in the null model.
python run_baseline.py

# Subsequent phases reuse Phase 1 caches and run in minutes
python run_phase2.py        # rolling σ time series + lead-lag tests
python run_phase3.py        # partial-correlation network comparison
python run_phase4.py        # Hawkes-process null comparison
python run_phase5.py        # same-day propagation, scaling, heatmap
                            #  (~30 min if 100-shuffle null is included)

# After Phases 1, 3, 4, 5 are done, generate alternative-axis figures
python make_linear_plots.py
```

If anything blows up, the first thing to check is which Python you're
using. The pipeline was developed against Python 3.14 in `.venv/`. The
parquet files in `data/` were written by that interpreter and may not
be readable from a different pyarrow version. If you see
"Repetition level histogram size mismatch", you're on the wrong
Python.

All scripts are deterministic; `RANDOM_SEED = 42` is set in
`src/config.py` and rerunning produces bit-identical numbers.

---

## The five phases in plain English

Each phase answers a different question. They're not redundant — each
one closes off a different alternative explanation a sceptical reader
would raise.

### Phase 1 — Is there even anything heavy-tailed here?

Pulls 503 S&P 500 stocks' daily prices from yfinance, computes log
returns, residualises against (market + GICS sector + SMB + HML),
flags days where each stock's residual exceeds 2σ on a 60-day trailing
window, builds a rolling 60-day Pearson correlation network at
|corr| > 0.4, and detects avalanches by flooding activations across
network neighbours from one trading day to the next.

Then runs a **100-shuffle null model**: independently shuffles each
stock's residual time series and reruns the whole pipeline 100 times.

*Result.* Empirical max avalanche size **485 stocks**; null mean **16.6
stocks** (p ≈ 0). The detector is not manufacturing avalanches from
noise — cross-stock temporal coupling is real.

### Phase 2 — Can σ predict crashes?

Different question. Phases 1, 3, 4, 5 all defend "the SOC story is
real." Phase 2 asks "*can we use it for anything?*"

Computes σ in trailing 60/120/252-day windows. Overlays six canonical
crisis dates. Runs lead-lag correlations and an **incremental
predictive regression** with HAC standard errors:
`fwd_vol = a + b·lagged_vol + c·σ + ε`.

*Result.* The naive Pearson r = 0.43 with forward 21-day vol vanishes
once you control for trailing vol: σ adds only +0.02 R² and is not
significant (p_HAC > 0.10) at any horizon. Of six canonical crises,
only two register a higher pre-event σ — and both are confounded by the
lookback overlapping the actual selloff.

This is the **honest negative result**. σ is a contemporaneous
regime indicator, not a forward predictor of vol beyond what trailing
vol already provides. Reporting it makes the rest of the paper more
credible.

### Phase 3 — Is the network real, or are these spurious factor edges?

The Pearson correlation network used in Phase 1 contains a lot of
*indirect* edges: two stocks that don't really influence each other
will still correlate if they share a common factor. A sceptic could
argue the avalanches are just common-factor co-movement, not network
contagion.

Phase 3 replaces the rolling correlation with a **partial-correlation
network** via LedoitWolf shrinkage of the precision matrix. Partial
correlations only fire on *direct* conditional dependencies. Then
reruns the avalanche pipeline.

*Result.* At τ_partial = 0.03 (matched edge density), σ = 0.509 —
identical to Phase 1's 0.499. The heavy tail is reproduced. The SOC
pattern survives factor deconfounding.

### Phase 4 — Do coordinated stock identities matter?

A different sceptical attack: maybe what looks like network propagation
is just temporal clustering. Hawkes self-excitement could explain why
bursty days cluster, and once you have an active day, *which* stocks
activate could be irrelevant — just the count and the empirical
network.

Phase 4 fits a univariate exponential Hawkes process to the active-day
timing. (`tick` 0.8 is broken on Python 3.14; the implementation uses
manual MLE + Ogata thinning per `agents.md` §4.1 Option 2.) Then
simulates 200 paths. On each simulated event day, the count is drawn
from the empirical "stocks per active day" distribution; the *which*
stocks come from per-stock empirical frequencies, with no cross-stock
coordination. Avalanche detection runs against the **empirical**
network (cached so all 200 sims complete in ~8 minutes).

*Result.* For sizes 1–50 the simulated and empirical CCDFs overlap
exactly — the network alone handles moderate avalanches. Above
~70 they diverge sharply. Empirical max 485 vs simulated max mean
277; **0 of 200 simulations** matched the empirical extreme. Cross-
stock identity coordination matters above and beyond temporal
clustering plus marginal stock rates.

### Phase 5 — Is this robust to detector resolution and parameter choice?

By the end of Phase 4, two weaknesses remained:

- σ ≈ 0.5 was awkwardly subcritical. SOC needs σ near 1.
- The marginal size distribution wasn't statistically distinguishable
  from a lognormal (LR test p = 0.66).

Phase 5 addresses both:

1. **Same-day propagation.** The Phase 1 detector only counts a stock
   activation as secondary if its network neighbour fired *yesterday*.
   With daily data, any cascade that completes within one trading day
   collapses into many simultaneous primary activations. The Phase 5
   detector adds within-day component-finding (gated behind a flag, so
   Phases 2/3/4 are unchanged): inside the same-day activation
   subgraph, exactly one canonical seed is primary and the rest are
   secondary. This corrects the daily-resolution bias.
2. **Crackling-noise size–duration scaling** (Sethna et al. 2001). SOC
   universality classes predict ⟨S | T⟩ ~ T^γ for specific γ; a
   lognormal process produces no such joint scaling. Bin avalanches by
   duration, regress mean log size on log duration, get γ. Run on three
   independent avalanche sets to test universality directly.
3. **(k, τ) heatmap.** Visualise σ and α over the 16-cell parameter
   grid to show that the result isn't tuned.

*Result.*
- Same-day: σ = **0.953 [0.795, 1.140]** — critical line σ = 1 is
  inside the 95 % CI.
- γ = 1.67 / 1.88 / 1.95 across Phase 1 / Phase 5 / Phase 3 avalanche
  sets — statistically consistent, universality argument secured.
- Heatmap shows a clean diagonal critical manifold; α stays in
  1.88–2.40 across the central region of the (k, τ) grid.

---

## Where to look in the outputs

Headline figures live in `outputs/figures/`. The most important ones,
in order:

```
kt_heatmap.png                       Phase 5 — σ and α over the (k, τ) grid
size_duration_scaling.png            Phase 5 — γ stability across 3 avalanche sets
ccdf_size_residuals_sameday.png      Phase 5 — heavy-tailed CCDF with α=1.90 fit
empirical_vs_null_sameday.png        Phase 5 — empirical vs 100-shuffle null
empirical_vs_null_sameday_natx.png   same, but natural x-axis (1–500)

ccdf_comparison_phase3.png           Phase 3 — raw vs partial correlation
ccdf_empirical_vs_hawkes.png         Phase 4 — empirical vs Hawkes-Poisson sims
rolling_branching_ratio.png          Phase 2 — σ_60/120/252 with crisis vlines

ccdf_size_raw_returns.png            Methodology — what happens without
                                        factor adjustment (σ = 5.6, supercritical)
```

The `_natx` variants render the same comparison with a natural
1–500 x-axis (semi-log) instead of log-log. They're more visceral for
the "empirical reaches 485, null dies at 30" story; the log-log
versions are better for reading off slopes.

Numerical summaries live in `outputs/tables/`:

```
baseline_summary.csv                 Phase 1 (k=2, τ=0.4) σ, α, fit stats
robustness_sweep.csv                 Phase 1 across 16 (k, τ) cells
null_summary.json                    Phase 1.8 null-model headline numbers
phase2_summary.json + lead_lag.csv + incremental_predictive.csv + precrisis_test.csv
phase3_summary.csv + phase3_summary.json
phase4_summary.json
baseline_summary_sameday.csv         Phase 5 same-day baseline
robustness_sweep_sameday.csv         Phase 5 across 16 cells
null_summary_sameday.json            Phase 5 null-model
scaling_exponents.csv                Phase 5 γ for each avalanche set
ccdf_powerlaw_fits.csv               Phase 5 — y = α/x^β fits per avalanche set
```

`agents.md` has the full technical spec, all five phase result
sections, every caveat, and the decision rationale for every
parameter. If you have a specific question this README doesn't answer,
look there.

---

## Repo layout

```
.
├── agents.md                  # full spec + complete results writeup
├── README.md                  # this file
├── requirements.txt
├── run_baseline.py            # Phase 1 driver
├── run_phase2.py              # Phase 2 driver
├── run_phase3.py              # Phase 3 driver
├── run_phase4.py              # Phase 4 driver
├── run_phase5.py              # Phase 5 driver
├── make_linear_plots.py       # natural-x companion figures
├── smoke_test.py              # synthetic-data correctness check
├── src/
│   ├── config.py              # paths, constants, RANDOM_SEED
│   ├── data_acquisition.py    # yfinance, Wikipedia, FF factors, OLS residuals
│   ├── activation.py          # |residual| > k·σ activation rule
│   ├── network.py             # rolling correlation network + cached snapshots
│   ├── partial_corr.py        # LedoitWolf partial-correlation network (Phase 3)
│   ├── avalanche.py           # BFS detector with optional same-day rule
│   ├── statistics.py          # power-law fit, branching-ratio bootstrap
│   ├── null_model.py          # per-stock time-shuffle null
│   ├── rolling_branching.py   # Phase 2 σ time series, lead-lag, regression
│   ├── hawkes.py              # Phase 4: Hawkes MLE, Ogata thinning, sims
│   ├── scaling.py             # Phase 5: size-duration scaling
│   └── plotting.py            # all figure helpers
├── data/                      # cached parquet/csv/pickle (built on first run)
├── outputs/
│   ├── figures/               # paper figures
│   └── tables/                # summary CSVs and JSONs
└── notebooks/                 # placeholder; analysis lives in run_*.py
```

The avalanche detector (`src/avalanche.py`) is intentionally
network-agnostic. It consumes anything with a
`neighbors_on(date) → dict[ticker, set[ticker]]` method. That's why
the partial-correlation network (Phase 3) drops in without touching
the detection logic.

---

## What this project is *not*

A few honest limitations to flag up front:

- **Survivorship-biased universe.** We use the *current* S&P 500
  constituent list. Stocks that were dropped (bankruptcies, mergers)
  are missing. The convention in the spec is to acknowledge and
  proceed; we follow it.
- **σ is not a crash predictor.** The Phase 2 incremental regression
  killed that hypothesis. If you came here for a leading indicator,
  this isn't one.
- **Daily resolution is the floor.** Same-day propagation in Phase 5
  recovers the σ ≈ 1 reading that intraday data would presumably
  produce directly, but we cannot verify that without the intraday
  data we don't have.
- **Phase 4 Hawkes degenerates to Poisson at the day level.** 5,197 of
  5,281 trading days have at least one activation, so there's nothing
  for self-excitement to fit; the model collapses to Poisson with rate
  ≈ 1. This is correctly diagnosed by the fit (α = 0). The simulation
  still serves as a clean coordinated-identity null.

---

## Reproducibility

Every random draw in the pipeline is seeded by
`RANDOM_SEED = 42` in `src/config.py`. Reruns with the same Python
environment produce bit-identical numerical outputs. If you change
the seed and want to verify, the headline numbers — σ, α, γ, the
null p-values, the scaling exponents — should remain in their
documented confidence intervals across seeds; that's part of what
the multiple-shuffle null model and the bootstrap CIs are for.

If you re-run from scratch and the numbers in this README disagree
with what you compute, *first* check that you're on Python 3.14
inside `.venv/`. The pyarrow / pandas version pinned by the venv
matters for binary parquet compatibility.
