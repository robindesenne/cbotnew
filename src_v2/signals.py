from __future__ import annotations

import numpy as np
import pandas as pd


def primary_setups_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    trend = (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])
    pullback = (df["close"] < df["ema_20"] * 1.01) & (df["close"] > df["ema_50"] * 0.98)
    rebound = (df["close"] > df["open"]) & (df["ret_1"] > 0)
    out["setup_pullback_v2"] = (trend & pullback & rebound).astype(int)

    breakout = (df["breakout_up"] == 1) & (df["rel_volume"] > 1.1)
    low_chop = df["compression_lookback"] < 0.95
    out["setup_breakout_v2"] = (breakout & low_chop.shift(1).astype("boolean").fillna(False)).astype(int)

    out["setup_any"] = ((out["setup_pullback_v2"] + out["setup_breakout_v2"]) > 0).astype(int)
    return out


def dynamic_threshold(proba_hist: pd.Series, base: float = 0.62) -> float:
    if len(proba_hist) < 100:
        return base
    p80 = float(np.nanpercentile(proba_hist.dropna().values, 80))
    return max(base, min(0.9, p80))
