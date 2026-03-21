from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()
import time

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d", retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            keep = ["Open", "High", "Low", "Close", "Volume"]
            df = df[keep].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            if "RateLimit" in type(e).__name__:
                wait = 60 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s... (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed to fetch {ticker} after {retries} retries")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def add_features(price_df: pd.DataFrame, sentiment_daily: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, List[str]]:
    df = price_df.copy().sort_index()

    df["ret_1d"] = df["Close"].pct_change()
    df["log_ret_1d"] = np.log(df["Close"]).diff()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    df["vol_chg_1d"] = df["Volume"].pct_change()
    df["vol_z_20d"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    df["rsi_14"] = _rsi(df["Close"], period=14)
    macd, macd_signal, macd_hist = _macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    for lag in (1, 2, 3, 5):
        df[f"ret_lag_{lag}"] = df["ret_1d"].shift(lag)
        df[f"vol_chg_lag_{lag}"] = df["vol_chg_1d"].shift(lag)

    if sentiment_daily is None:
        df["sentiment"] = 0.0
    else:
        df["sentiment"] = sentiment_daily.reindex(df.index).astype(float).fillna(0.0)

    df["target_next_ret_1d"] = df["ret_1d"].shift(-1)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ret_1d",
        "log_ret_1d",
        "vol_20d",
        "vol_chg_1d",
        "vol_z_20d",
        "sma_10",
        "sma_20",
        "ema_10",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "sentiment",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_5",
        "vol_chg_lag_1",
        "vol_chg_lag_2",
        "vol_chg_lag_3",
        "vol_chg_lag_5",
    ]

    df = df.dropna(subset=feature_cols + ["target_next_ret_1d"]).copy()
    return df, feature_cols


def train_and_persist_model(ticker: str, out_path: str = "model.joblib") -> str:
    import joblib
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `scikit-learn`. Install it (e.g. `pip install scikit-learn`) to train the model."
        ) from exc

    prices = fetch_ohlcv(ticker, period="5y")
    feat_df, feature_cols = add_features(prices)
    X = feat_df[feature_cols]
    y = feat_df["target_next_ret_1d"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    model.fit(X, y)

    joblib.dump({"model": model, "feature_cols": feature_cols}, out_path)
    return out_path


def alpaca_paper_trade_once(ticker: str, model_path: str, qty: int = 1, threshold: float = 0.0):
    """One-shot decision + Alpaca *paper* market order.

    - Loads a saved sklearn model bundle (`joblib`).
    - Fetches latest daily bar history from yfinance (demo).
    - Computes features for the latest row and predicts next-day return.
    - If `pred > threshold` -> BUY else SELL.
    """
    import joblib

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `alpaca-py`. Install it (e.g. `pip install alpaca-py`) to use paper trading."
        ) from exc

    key = os.getenv('ALPACA_API_KEY_ID')
    secret = os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` environment variables first.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    prices = fetch_ohlcv(ticker, period="6mo")
    feat_df, _ = add_features(prices)
    latest_X = feat_df[feature_cols].iloc[[-1]]
    pred_next_ret = float(model.predict(latest_X)[0])

    side = "buy" if pred_next_ret > threshold else "sell"
    client = TradingClient(key, secret, paper=True)
    order = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    submitted = client.submit_order(order)
    return {
        "ticker": ticker,
        "pred_next_ret": pred_next_ret,
        "threshold": threshold,
        "decision": side,
        "qty": qty,
        "submitted_order": submitted,
    }
