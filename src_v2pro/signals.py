from __future__ import annotations

import pandas as pd


def primary_setups_pro(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # A bit more permissive to keep enough event flow for statistical robustness
    trend = (df["ema_20"] > df["ema_50"]) & (df["dist_ema_50"] > -0.01)
    pullback = (df["close"] <= df["ema_20"] * 1.02) & (df["close"] >= df["ema_50"] * 0.975)
    rebound = (df["close"] > df["open"]) | (df["ret_1"] > 0)

    breakout = (df["breakout_up"] == 1) & (df["rel_volume"] > 0.95)
    vol_ok = (df["rv_24"] > df["rv_24"].rolling(120).quantile(0.15))

    out["setup_trend_pullback"] = (trend & pullback & rebound).astype(int)
    out["setup_breakout"] = (breakout & vol_ok.fillna(False)).astype(int)

    # Optional continuation block to increase frequency in strong trends
    continuation = (df["ema_20"] > df["ema_50"]) & (df["ret_3"] > 0) & (df["rel_volume"] > 0.8)
    out["setup_continuation"] = continuation.astype(int)

    out["setup_any"] = ((out[["setup_trend_pullback", "setup_breakout", "setup_continuation"]].sum(axis=1)) > 0).astype(int)
    return out
