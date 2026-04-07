from __future__ import annotations

import numpy as np
import pandas as pd


def regime_rules(df: pd.DataFrame) -> pd.Series:
    bull = (df["ema_50"] > df["ema_200"]) & (df["dist_ema_200"] > 0) & (df["rv_24"] < df["rv_24"].rolling(200).quantile(0.85))
    bear = (df["ema_50"] < df["ema_200"]) & (df["dist_ema_200"] < 0)
    reg = pd.Series("range", index=df.index)
    reg[bull] = "bull"
    reg[bear] = "bear"
    return reg


def regime_transition_prob(regime: pd.Series, lookback: int = 200) -> pd.DataFrame:
    """Markov empirique léger: probabilité de rester en bull/bear/range (version vectorisée)."""
    states = ["bull", "bear", "range"]
    prev = regime.shift(1)
    out = pd.DataFrame(index=regime.index, dtype=float)

    for s in states:
        prev_s = (prev == s).astype(float)
        stay_s = ((prev == s) & (regime == s)).astype(float)

        denom = prev_s.rolling(lookback, min_periods=1).sum()
        numer = stay_s.rolling(lookback, min_periods=1).sum()

        p = numer / denom.replace(0, np.nan)
        out[f"p_stay_{s}"] = p.fillna(0.33).astype(float)

    return out
