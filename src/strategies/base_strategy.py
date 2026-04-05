from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..features import add_features
from ..labels import fixed_horizon_label, triple_barrier_label
from ..regime import regime_rules, regime_transition_prob
from ..signals import primary_setups
from ..execution import simulate_spot_long_only


@dataclass
class StrategyContext:
    symbol: str = "SOLUSDT"
    interval: str = "1h"
    horizon_bars: int = 24
    config: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Base abstraite pour toutes les stratégies quant.

    Contrat de compatibilité backtest_engine:
    - generate_signals(...) doit renvoyer un DataFrame contenant au minimum:
      - trade_flag (int 0/1)
      - meta_proba (float [0,1], optionnel mais recommandé)
    - les index/timestamps doivent rester alignés avec le DataFrame d'entrée.
    """

    name: str = "base"
    version: str = "0.0.0"
    family: str = "generic"
    default_params: Dict[str, Any] = {}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = {**self.default_params, **(params or {})}

    def prepare_dataset(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        """
        Pipeline standardisé features + setup + labels + régime.
        Fast-path: si dataset déjà enrichi (pipeline SOLUSDT), éviter tout recalcul.
        """
        out = df.copy()

        # Fast path: already engineered by backtest.solusdt_pipeline
        required = {"setup_any", "label", "label_tb", "label_fh", "event_end_idx", "regime"}
        if not required.issubset(set(out.columns)):
            out = add_features(out)

            setups = primary_setups(out)
            out = pd.concat([out, setups], axis=1)

            h = int(ctx.horizon_bars)
            out["label_tb"] = triple_barrier_label(out, horizon=h, up_mult=1.5, dn_mult=1.0)
            out["label_fh"] = fixed_horizon_label(out, horizon=h, threshold=0.0)
            out["label"] = out["label_tb"].fillna(out["label_fh"])
            out["event_end_idx"] = np.minimum(np.arange(len(out)) + h, len(out) - 1)

            out["regime"] = regime_rules(out)
            trans = regime_transition_prob(out["regime"]).fillna(0.33)
            out = pd.concat([out, trans], axis=1)

        if "trade_flag" not in out.columns:
            out["trade_flag"] = out.get("setup_any", 0).fillna(0).astype(int)
        if "meta_proba" not in out.columns:
            out["meta_proba"] = 0.0

        return out

    def run_execution(self, signal_df: pd.DataFrame, ctx: StrategyContext):
        """Bridge execution commun pour rester compatible avec execution.py."""
        cfg = ctx.config or {}
        return simulate_spot_long_only(signal_df, cfg)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        """Retourne un DataFrame aligné contenant au minimum trade_flag (+ meta_proba recommandé)."""
        raise NotImplementedError
