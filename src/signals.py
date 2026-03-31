from __future__ import annotations

import pandas as pd


def primary_setups(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # 1) trend pullback
    trend = (df["ema_50"] > df["ema_200"]) & (df["dist_ema_200"] > 0)
    pullback = (df["close"] <= df["ema_20"] * 1.005) & (df["close"] >= df["ema_50"] * 0.99)
    rebound = (df["close"] > df["open"]) & (df["close"] > df["high"].shift(1))
    out["setup_trend_pullback"] = (trend & pullback & rebound).astype(int)

    # 2) breakout with vol expansion
    compression = df["range_pct"] < df["range_pct"].rolling(50).quantile(0.25)
    breakout = df["breakout_up"] == 1
    vol_ok = df["rel_volume"] > 1.0
    out["setup_breakout"] = (compression.shift(1).astype("boolean").fillna(False) & breakout & vol_ok).astype(int)

    # 3) continuation after compression
    trend2 = (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])
    cont = (df["ret_3"] > 0) & (df["ret_1"] > 0)
    out["setup_continuation"] = (trend2 & compression.shift(2).astype("boolean").fillna(False) & cont).astype(int)

    out["setup_any"] = (out.sum(axis=1) > 0).astype(int)
    return out


def final_trade_flag(df: pd.DataFrame, proba_col: str = "meta_proba", threshold: float = 0.62) -> pd.Series:
    return ((df["setup_any"] == 1) & (df[proba_col] >= threshold)).astype(int)
