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

    # MACD-like
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Trix-like
    trix_1 = close.ewm(span=15, adjust=False).mean()
    trix_2 = trix_1.ewm(span=15, adjust=False).mean()
    trix_3 = trix_2.ewm(span=15, adjust=False).mean()
    df["trix_15"] = trix_3.pct_change()

    # Bollinger-like
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_mid_20"] = bb_mid
    df["bb_up_20_2"] = bb_mid + 2.0 * bb_std
    df["bb_dn_20_2"] = bb_mid - 2.0 * bb_std
    df["bb_width_20"] = (df["bb_up_20_2"] - df["bb_dn_20_2"]) / close.replace(0, np.nan)

    # Stochastic-like
    ll_14 = df["low"].rolling(14).min()
    hh_14 = df["high"].rolling(14).max()
    df["stoch_k_14"] = 100 * (close - ll_14) / (hh_14 - ll_14).replace(0, np.nan)
    df["stoch_d_3"] = df["stoch_k_14"].rolling(3).mean()

    # VWAP rolling proxy
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_96"] = (tp * df["volume"]).rolling(96).sum() / df["volume"].rolling(96).sum().replace(0, np.nan)
    df["dist_vwap_96"] = close / df["vwap_96"] - 1

    # simple zscores
    for c in ["range_pct", "atr_pct", "rv_24", "rel_volume", "bb_width_20", "dist_vwap_96"]:
        mu = df[c].rolling(100).mean()
        sd = df[c].rolling(100).std().replace(0, np.nan)
        df[f"z_{c}"] = (df[c] - mu) / sd

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"ts", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
