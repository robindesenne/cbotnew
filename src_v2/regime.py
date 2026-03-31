from __future__ import annotations

import pandas as pd


def detect_regime_v2(df: pd.DataFrame) -> pd.Series:
    trend_up = (df["ema_50"] > df["ema_200"]) & (df["trend_strength"] > 0.01)
    trend_dn = (df["ema_50"] < df["ema_200"]) & (df["trend_strength"] > 0.01)
    chop = (df["compression_lookback"] < 0.9) | (df["vol_ratio_30_90"] < 0.85)

    regime = pd.Series("neutral", index=df.index)
    regime[trend_up & ~chop] = "bull"
    regime[trend_dn & ~chop] = "bear"
    regime[chop] = "chop"
    return regime


def regime_threshold_adjustment(regime: str) -> float:
    # dynamic threshold adjustment by regime
    if regime == "bull":
        return -0.03
    if regime == "bear":
        return +0.08
    if regime == "chop":
        return +0.12
    return 0.0
