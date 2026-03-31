from __future__ import annotations

import pandas as pd


def regime_rules(df: pd.DataFrame) -> pd.Series:
    bull = (df["ema_50"] > df["ema_200"]) & (df["dist_ema_200"] > 0) & (df["rv_24"] < df["rv_24"].rolling(200).quantile(0.85))
    bear = (df["ema_50"] < df["ema_200"]) & (df["dist_ema_200"] < 0)
    reg = pd.Series("range", index=df.index)
    reg[bull] = "bull"
    reg[bear] = "bear"
    return reg


def regime_transition_prob(regime: pd.Series, lookback: int = 200) -> pd.DataFrame:
    """Markov empirique léger: probabilité de rester en bull/bear/range."""
    states = ["bull", "bear", "range"]
    out = pd.DataFrame(index=regime.index, columns=[f"p_stay_{s}" for s in states], dtype=float)
    for i in range(lookback, len(regime)):
        w = regime.iloc[i - lookback:i]
        prev = w.shift(1)
        for s in states:
            mask = prev == s
            if mask.sum() == 0:
                out.iloc[i, out.columns.get_loc(f"p_stay_{s}")] = 0.33
            else:
                out.iloc[i, out.columns.get_loc(f"p_stay_{s}")] = (w[mask] == s).mean()
    return out
