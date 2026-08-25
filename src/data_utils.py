"""
Data loading, preprocessing, and sliding-window sequence preparation
for the stock price prediction project.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def download_stock_data(
    ticker: str = "AMZN",
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Download daily OHLCV data for `ticker` using yfinance.

    Caches the result to `cache_dir/<ticker>.csv` so repeated runs don't
    re-download. If there is no internet connection (or yfinance fails),
    falls back to a locally generated synthetic random-walk price series
    so the rest of the pipeline can still be exercised offline.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    try:
        import yfinance as yf

        df = yf.download(ticker,start=start,end=end,progress=False,multi_level_index=False)
        if df.empty:
            raise ValueError("yfinance returned an empty dataframe")
        df.to_csv(cache_path)
        return df
    except Exception as exc:  # noqa: BLE001 - broad on purpose, this is a fallback path
        print(f"[data_utils] Could not download '{ticker}' via yfinance ({exc}). "
              f"Falling back to synthetic data for offline use.")
        return _generate_synthetic_price_series(cache_path)


def _generate_synthetic_price_series(
    cache_path: str,
    n_days: int = 3000,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic geometric-random-walk 'Close' price series."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(loc=0.0004, scale=0.02, size=n_days)
    prices = start_price * np.cumprod(1 + daily_returns)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    df = pd.DataFrame({"Close": prices}, index=dates)
    df.index.name = "Date"
    df.to_csv(cache_path)
    return df


def create_sliding_windows(series: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sequence pairs from a 1D price series using a sliding window.

    X[i] = series[i : i+lookback]
    y[i] = series[i+lookback]        (the next value after the window)
    """
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback])
    return np.array(X), np.array(y)


def prepare_data(
    df: pd.DataFrame,
    lookback: int = 20,
    train_split: float = 0.8,
    price_col: str = "Close",
):
    """
    Full preprocessing pipeline: extract the price column, scale to [-1, 1],
    build sliding-window sequences, and split chronologically into
    train/test sets (no shuffling — this is time series data).

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Shapes: X_* is (n_samples, lookback, 1), y_* is (n_samples, 1)
    scaler : MinMaxScaler
        Fitted scaler, needed to inverse-transform predictions back to price units.
    """
    prices = df[[price_col]].values.astype("float32")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(prices).flatten()

    X, y = create_sliding_windows(scaled, lookback)
    X = X.reshape(-1, lookback, 1)
    y = y.reshape(-1, 1)

    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test, scaler
