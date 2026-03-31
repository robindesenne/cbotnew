from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import add_features as add_features_v1, feature_columns as feature_columns_v1


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features_v1(df)
    # V2 additions (no external APIs yet, OHLCV-only robust)
    df["trend_strength"] = (df["ema_50"] / df["ema_200"] - 1.0).abs()
    df["vol_ratio_30_90"] = df["rv_24"].rolling(30).mean() / df["rv_24"].rolling(90).mean()
    df["compression_lookback"] = (df["range_pct"].rolling(20).mean() / df["range_pct"].rolling(100).mean())
    df["hurst_proxy"] = df["ret_1"].rolling(120).apply(lambda x: np.sign(np.corrcoef(np.arange(len(x)), np.cumsum(np.nan_to_num(x)))[0,1]) if np.isfinite(x).all() else np.nan, raw=False)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = feature_columns_v1(df)
    drop = {"label", "label_tb", "label_fh", "meta_proba", "trade_flag", "setup_any", "regime"}
    return [c for c in cols if c not in drop]
