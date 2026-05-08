"""
Phase 1.1-1.2: Data acquisition and factor adjustment.

Steps:
  1. Pull current S&P 500 constituents from Wikipedia (with GICS sector).
  2. Download daily adjusted close prices via yfinance, 2005-2025.
  3. Compute log returns.
  4. Download Fama-French daily factors (SMB, HML, Mkt-RF, RF).
  5. For each stock, regress log return on:
        market return (^GSPC), GICS-sector equal-weighted return,
        SMB, HML
     and keep the residual as the factor-adjusted return.
  6. Cache prices, returns, residuals, sector map, factors as parquet/csv.

All public functions are idempotent: they return cached results when
available and only refetch/recompute when force=True.
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf

from .config import DATA_DIR, END_DATE, START_DATE


SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FF_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


# ---------------------------------------------------------------------------
# S&P 500 universe
# ---------------------------------------------------------------------------
def get_sp500_universe(force: bool = False) -> pd.DataFrame:
    """Return DataFrame with columns ['ticker', 'sector']."""
    cache = DATA_DIR / "sp500_universe.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    # Wikipedia blocks default urllib UA; fetch via requests with a real UA.
    resp = requests.get(
        SP500_WIKI_URL,
        headers={"User-Agent": "Mozilla/5.0 (research script)"},
        timeout=30,
    )
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    df = pd.DataFrame(
        {
            "ticker": table["Symbol"].str.replace(".", "-", regex=False),
            "sector": table["GICS Sector"],
        }
    )
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Price download
# ---------------------------------------------------------------------------
def _download_batch(
    tickers: Iterable[str], start: str, end: str, retries: int = 3, sleep: float = 2.0
) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                list(tickers),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"]
            else:
                df = df[["Close"]]
                df.columns = list(tickers)
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"yfinance batch failed after {retries} retries") from last_err


def download_prices(
    tickers: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
    batch_size: int = 100,
    force: bool = False,
) -> pd.DataFrame:
    """Daily adjusted close prices, columns = tickers, index = trading days."""
    cache = DATA_DIR / "raw_prices.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  downloading {i:>4d}-{i + len(batch):>4d} of {len(tickers)}")
        df = _download_batch(batch, start, end)
        frames.append(df)
        time.sleep(1.0)  # be polite

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()
    prices.to_parquet(cache)
    return prices


def download_market_index(
    start: str = START_DATE, end: str = END_DATE, force: bool = False
) -> pd.Series:
    """S&P 500 index daily close (^GSPC), as a Series."""
    cache = DATA_DIR / "market_index.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache).iloc[:, 0]

    idx = yf.download(
        "^GSPC",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(idx.columns, pd.MultiIndex):
        idx = idx["Close"]
    s = idx["^GSPC"] if "^GSPC" in idx.columns else idx.iloc[:, 0]
    s.name = "GSPC"
    s.to_frame().to_parquet(cache)
    return s


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ---------------------------------------------------------------------------
# Fama-French
# ---------------------------------------------------------------------------
def download_ff_factors(force: bool = False) -> pd.DataFrame:
    """Daily FF factors (Mkt-RF, SMB, HML, RF) in decimal units, indexed by date."""
    cache = DATA_DIR / "ff_factors.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df

    resp = requests.get(FF_DAILY_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(name) as f:
            text = f.read().decode("latin-1")

    # The CSV has a header preamble and an "Annual Factors" trailer.
    # Find the first line that looks like YYYYMMDD,...
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        first = line.split(",")[0].strip()
        if len(first) == 8 and first.isdigit():
            start = i
            break
    if start is None:
        raise RuntimeError("Could not locate data start in FF CSV")

    end = len(lines)
    for j in range(start, len(lines)):
        first = lines[j].split(",")[0].strip()
        if not (len(first) == 8 and first.isdigit()):
            end = j
            break

    csv_block = "Date,Mkt-RF,SMB,HML,RF\n" + "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(csv_block))
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").sort_index()
    df = df.astype(float) / 100.0  # FF reports percent
    df.to_csv(cache)
    return df


# ---------------------------------------------------------------------------
# Sector returns
# ---------------------------------------------------------------------------
def compute_sector_returns(
    log_returns: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    """Equal-weighted daily mean log return per GICS sector."""
    sector_map = universe.set_index("ticker")["sector"]
    common = [t for t in log_returns.columns if t in sector_map.index]
    by_sector: dict[str, pd.Series] = {}
    for sector, group in sector_map.loc[common].groupby(sector_map.loc[common]):
        cols = group.index.tolist()
        by_sector[sector] = log_returns[cols].mean(axis=1)
    return pd.DataFrame(by_sector)


# ---------------------------------------------------------------------------
# Residuals via OLS
# ---------------------------------------------------------------------------
def compute_residuals(
    log_returns: pd.DataFrame,
    market_ret: pd.Series,
    sector_ret: pd.DataFrame,
    ff: pd.DataFrame,
    universe: pd.DataFrame,
    min_obs: int = 250,
) -> pd.DataFrame:
    """
    For each stock: residuals from
        r_i = a + b_m * r_mkt + b_s * r_sector + b_smb * SMB + b_hml * HML + e
    Returns a DataFrame aligned with log_returns (NaN where the regression
    couldn't be run for that stock).
    """
    sector_map = universe.set_index("ticker")["sector"].to_dict()

    common_idx = (
        log_returns.index.intersection(market_ret.index)
        .intersection(sector_ret.index)
        .intersection(ff.index)
    )
    lr = log_returns.loc[common_idx]
    mkt = market_ret.loc[common_idx]
    sec = sector_ret.loc[common_idx]
    smb = ff.loc[common_idx, "SMB"]
    hml = ff.loc[common_idx, "HML"]

    residuals = pd.DataFrame(index=common_idx, columns=lr.columns, dtype=float)

    for ticker in lr.columns:
        y = lr[ticker]
        sector = sector_map.get(ticker)
        if sector is None or sector not in sec.columns:
            continue
        X = pd.concat(
            [mkt.rename("mkt"), sec[sector].rename("sec"), smb, hml],
            axis=1,
        )
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(df) < min_obs:
            continue
        Xc = sm.add_constant(df.drop(columns=["y"]), has_constant="add")
        try:
            res = sm.OLS(df["y"], Xc).fit()
        except Exception:  # noqa: BLE001
            continue
        residuals.loc[df.index, ticker] = res.resid.values

    return residuals


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------
def build_dataset(force: bool = False) -> dict[str, pd.DataFrame]:
    """Run the full data step. Returns dict of cached frames."""
    universe = get_sp500_universe(force=force)
    print(f"Universe: {len(universe)} tickers")

    prices = download_prices(universe["ticker"].tolist(), force=force)
    print(f"Prices: {prices.shape}")

    log_returns = compute_log_returns(prices)
    log_returns.to_parquet(DATA_DIR / "log_returns.parquet")
    print(f"Log returns: {log_returns.shape}")

    market = download_market_index(force=force)
    ff = download_ff_factors(force=force)
    sector_ret = compute_sector_returns(log_returns, universe)
    sector_ret.to_parquet(DATA_DIR / "sector_returns.parquet")

    residuals = compute_residuals(log_returns, market, sector_ret, ff, universe)
    residuals.to_parquet(DATA_DIR / "residuals.parquet")
    print(f"Residuals: {residuals.shape}")

    return {
        "universe": universe,
        "prices": prices,
        "log_returns": log_returns,
        "market": market,
        "ff": ff,
        "sector_returns": sector_ret,
        "residuals": residuals,
    }


if __name__ == "__main__":
    build_dataset()
