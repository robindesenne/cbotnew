from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..data_loader import LoadSpec, load_ohlcv
from ..features import add_features
from ..labels import fixed_horizon_label, triple_barrier_label
from ..regime import regime_rules, regime_transition_prob
from ..signals import primary_setups
from .config import GLOBAL_CONFIG


@dataclass
class SolusdtPipelineSpec:
    interval: str = GLOBAL_CONFIG["solusdt"]["default_timeframe"]
    date_from: str = GLOBAL_CONFIG["solusdt"]["history_start"]
    date_to: str = GLOBAL_CONFIG["solusdt"]["history_end"]


def build_solusdt_dataset(root: Path, spec: SolusdtPipelineSpec) -> Tuple[pd.DataFrame, str]:
    sconf: Dict = GLOBAL_CONFIG["solusdt"]
    load_spec = LoadSpec(
        symbol=sconf["symbol"],
        interval=spec.interval,
        date_from=spec.date_from,
        date_to=spec.date_to,
        prefer_local=bool(sconf.get("prefer_local", True)),
        market_type=sconf.get("market_type", "spot"),
    )
    df, source = load_ohlcv(root, load_spec)

    df = add_features(df)
    setups = primary_setups(df)
    df = pd.concat([df, setups], axis=1)

    h = int(GLOBAL_CONFIG["label_horizon_bars"])
    df["label_tb"] = triple_barrier_label(df, horizon=h, up_mult=1.5, dn_mult=1.0)
    df["label_fh"] = fixed_horizon_label(df, horizon=h, threshold=0.0)
    df["label"] = df["label_tb"].fillna(df["label_fh"])
    df["event_end_idx"] = np.minimum(np.arange(len(df)) + h, len(df) - 1)

    df["regime"] = regime_rules(df)
    trans = regime_transition_prob(df["regime"]).fillna(0.33)
    df = pd.concat([df, trans], axis=1)

    return df, source
