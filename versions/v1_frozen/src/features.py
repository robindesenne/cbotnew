from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    # returns & momentum multi-horizon
    for h in [1, 3, 6, 12, 24, 48]:
        df[f"ret_{h}"] = close.pct_change(h)
        df[f"mom_{h}"] = close / close.shift(h) - 1

    # trend
    for n in [20, 50, 100, 200]:
        df[f"ema_{n}"] = close.ewm(span=n, adjust=False).mean()
        df[f"dist_ema_{n}"] = close / df[f"ema_{n}"] - 1
        df[f"slope_ema_{n}"] = df[f"ema_{n}"].pct_change(5)

    # vol & range
    df["atr_14"] = _atr(df, 14)
    df["atr_pct"] = df["atr_14"] / close
    df["rv_24"] = close.pct_change().rolling(24).std()
    df["rv_96"] = close.pct_change().rolling(96).std()
    df["range_pct"] = (df["high"] - df["low"]) / close.replace(0, np.nan)

    # volume
    df["vol_med_20"] = df["volume"].rolling(20).median()
    df["rel_volume"] = df["volume"] / df["vol_med_20"].replace(0, np.nan)
    df["vol_z_50"] = (df["volume"] - df["volume"].rolling(50).mean()) / df["volume"].rolling(50).std()

    # structure / candle
    df["close_in_bar"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["local_drawdown_48"] = close / close.rolling(48).max() - 1

    # breakout context
    df["hh_20"] = df["high"].rolling(20).max().shift(1)
    df["ll_20"] = df["low"].rolling(20).min().shift(1)
    df["breakout_up"] = (df["close"] > df["hh_20"]).astype(int)
    df["breakout_dn"] = (df["close"] < df["ll_20"]).astype(int)

    # oscillators baseline
    df["rsi_14"] = _rsi(close, 14)

    # simple zscores
    for c in ["range_pct", "atr_pct", "rv_24", "rel_volume"]:
        mu = df[c].rolling(100).mean()
        sd = df[c].rolling(100).std().replace(0, np.nan)
        df[f"z_{c}"] = (df[c] - mu) / sd

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"ts", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
