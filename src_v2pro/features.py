from __future__ import annotations

import pandas as pd

from src.features import add_features as add_features_base


# Strict allowlist to prevent accidental leakage through broad numeric selection
FEATURE_ALLOWLIST = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "mom_3", "mom_6", "mom_12", "mom_24",
    "dist_ema_20", "dist_ema_50", "dist_ema_200",
    "slope_ema_20", "slope_ema_50", "slope_200",
    "atr_pct", "rv_24", "rv_96", "range_pct",
    "rel_volume", "vol_z_50", "close_in_bar", "local_drawdown_48",
    "breakout_up", "breakout_dn", "rsi_14",
    "z_range_pct", "z_atr_pct", "z_rv_24", "z_rel_volume",
]


# backward compatibility if slope_200 doesn't exist in base pipeline
FALLBACK_MAP = {
    "slope_200": "slope_ema_200",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_features_base(df)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in FEATURE_ALLOWLIST:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        elif c in FALLBACK_MAP and FALLBACK_MAP[c] in df.columns and pd.api.types.is_numeric_dtype(df[FALLBACK_MAP[c]]):
            cols.append(FALLBACK_MAP[c])
    return sorted(set(cols))
