from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_horizon_label(df: pd.DataFrame, horizon: int = 24, threshold: float = 0.0) -> pd.Series:
    fut = df["close"].shift(-horizon) / df["close"] - 1
    return (fut > threshold).astype(int)


def triple_barrier_label(
    df: pd.DataFrame,
    horizon: int = 24,
    up_mult: float = 1.5,
    dn_mult: float = 1.0,
    atr_col: str = "atr_14",
) -> pd.Series:
    n = len(df)
    out = np.full(n, np.nan)
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = df[atr_col].values

    for i in range(n - horizon):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        up = c[i] + up_mult * atr[i]
        dn = c[i] - dn_mult * atr[i]
        label = 0
        for j in range(i + 1, i + horizon + 1):
            if h[j] >= up:
                label = 1
                break
            if l[j] <= dn:
                label = 0
                break
        out[i] = label
    return pd.Series(out, index=df.index)
