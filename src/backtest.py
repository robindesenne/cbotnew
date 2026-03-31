from __future__ import annotations

import json
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data_loader import LoadSpec, load_ohlcv
from .execution import simulate_spot_long_only
from .features import add_features, feature_columns
from .labels import fixed_horizon_label, triple_barrier_label
from .models import calibrate_model, choose_threshold, chronological_split, train_candidates
from .regime import regime_rules, regime_transition_prob
from .signals import final_trade_flag, primary_setups


def prepare_full_dataset(root: Path, symbol: str, interval: str, cfg: dict) -> tuple[pd.DataFrame, str]:
    """Prepare full local dataset with features/setups/labels once for reuse across overlapping periods."""
    spec = LoadSpec(symbol=symbol, interval=interval, date_from="2010-01-01", date_to="2100-01-01",
                    market_type=cfg.get("market", {}).get("type", "spot"))
    df, source = load_ohlcv(root, spec)
    df = add_features(df)
    setups = primary_setups(df)
    df = pd.concat([df, setups], axis=1)
    df["label_tb"] = triple_barrier_label(df, horizon=24, up_mult=1.5, dn_mult=1.0)
    df["label_fh"] = fixed_horizon_label(df, horizon=24, threshold=0.0)
    df["label"] = df["label_tb"].fillna(df["label_fh"])
    return df, source


def run_single_backtest(root: Path, symbol: str, interval: str, date_from: str, date_to: str, cash: float, prepared_df: Optional[pd.DataFrame]=None, prepared_source: Optional[str]=None):
    cfg = yaml.safe_load((root / "config.yaml").read_text())

    if prepared_df is None:
        spec = LoadSpec(symbol=symbol, interval=interval, date_from=date_from, date_to=date_to,
                        market_type=cfg.get("market", {}).get("type", "spot"))
        df, source = load_ohlcv(root, spec)
        df = add_features(df)
        setups = primary_setups(df)
        df = pd.concat([df, setups], axis=1)
        df["label_tb"] = triple_barrier_label(df, horizon=24, up_mult=1.5, dn_mult=1.0)
        df["label_fh"] = fixed_horizon_label(df, horizon=24, threshold=0.0)
        df["label"] = df["label_tb"].fillna(df["label_fh"])
    else:
        source = prepared_source or "prepared_cache"
        df = prepared_df.copy()
        d0 = pd.Timestamp(date_from, tz="UTC")
        d1 = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df.ts >= d0) & (df.ts < d1)].copy()

    # regime
    df["regime"] = regime_rules(df)
    trans = regime_transition_prob(df["regime"]).fillna(0.33)
    df = pd.concat([df, trans], axis=1)

    feats = feature_columns(df)
    ml = df.dropna(subset=feats + ["label", "setup_any"]).copy()
    ml = ml[ml["setup_any"] == 1].copy()

    model_name = "primary_only"
    valid_auc = None
    threshold = 0.5

    if len(ml) >= 500 and ml["label"].nunique() > 1:
        tr, va, te = chronological_split(ml)
        Xtr, ytr = tr[feats], tr["label"].astype(int)
        Xva, yva = va[feats], va["label"].astype(int)
        Xte, yte = te[feats], te["label"].astype(int)

        if len(Xtr) > 100 and len(Xva) > 50 and ytr.nunique() > 1 and yva.nunique() > 1:
            candidates = train_candidates(Xtr, ytr, Xva, yva)
            best = sorted(candidates, key=lambda x: x.auc, reverse=True)[0]
            model = calibrate_model(best.model, Xva, yva)

            te_proba = pd.Series(model.predict_proba(Xte)[:, 1], index=te.index)
            threshold = choose_threshold(te_proba, yte, min_precision=0.52)

            # apply model on full frame where setups exist
            df["meta_proba"] = np.nan
            valid_idx = ml.index
            df.loc[valid_idx, "meta_proba"] = model.predict_proba(ml[feats])[:, 1]
            df["meta_proba"] = df["meta_proba"].fillna(0.0)
            df["trade_flag"] = final_trade_flag(df, "meta_proba", threshold)

            model_name = best.name
            valid_auc = best.auc
        else:
            df["meta_proba"] = 1.0
            df["trade_flag"] = df["setup_any"].astype(int)
    else:
        df["meta_proba"] = 1.0
        df["trade_flag"] = df["setup_any"].astype(int)

    tdf, edf = simulate_spot_long_only(df, cfg)

    # metrics
    net_pnl = float(tdf["net"].sum()) if len(tdf) else 0.0
    fees_paid_total = float(tdf["fees"].sum()) if len(tdf) and "fees" in tdf.columns else 0.0

    # Buy&Hold net over selected backtest period (same costs model: entry+exit taker + slippage)
    if len(df) >= 2:
        entry_bh = float(df.iloc[0]["close"])
        exit_bh = float(df.iloc[-1]["close"])
        data_start_ts = str(df.iloc[0]["ts"])
        data_end_ts = str(df.iloc[-1]["ts"])
        bh_cost = (cfg["exchange"]["taker_fee"] * 2.0) + (cfg["exchange"]["slippage_bps"] / 10000.0 * 2.0)
        bh_return_gross_pct = (exit_bh / entry_bh - 1.0)
        bh_return_pct = bh_return_gross_pct - bh_cost
        bh_net_pnl = cash * bh_return_pct
        bh_final_equity = cash + bh_net_pnl
    else:
        bh_return_pct = 0.0
        bh_return_gross_pct = 0.0
        bh_net_pnl = 0.0
        bh_final_equity = cash
        data_start_ts = None
        data_end_ts = None
        entry_bh = None
        exit_bh = None
    wins = int((tdf["net"] > 0).sum()) if len(tdf) else 0
    losses = int((tdf["net"] <= 0).sum()) if len(tdf) else 0
    gp = float(tdf.loc[tdf["net"] > 0, "net"].sum()) if len(tdf) else 0.0
    gl = float(-tdf.loc[tdf["net"] <= 0, "net"].sum()) if len(tdf) else 0.0
    pf = (gp / gl) if gl > 0 else float("inf")
    max_dd = float(edf["drawdown"].max()) if len(edf) else 0.0

    rets = edf["equity"].pct_change().dropna() if len(edf) else pd.Series(dtype=float)
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * np.sqrt(365 * 24 * 4)) if len(rets) > 10 else 0.0

    summary = {
        "symbol": symbol,
        "interval": interval,
        "date_from": date_from,
        "date_to": date_to,
        "data_source": source,
        "model": model_name,
        "valid_auc": valid_auc,
        "threshold": threshold,
        "trades": int(len(tdf)),
        "fees_paid_total": fees_paid_total,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(tdf)) if len(tdf) else 0.0,
        "net_pnl": net_pnl,
        "return_pct": float(net_pnl / cash),
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "bh_return_pct": float(bh_return_pct),
        "bh_return_gross_pct": float(bh_return_gross_pct),
        "bh_net_pnl": float(bh_net_pnl),
        "bh_final_equity": float(bh_final_equity),
        "alpha_vs_bh_pct": float((net_pnl / cash) - bh_return_pct),
        "data_start_ts": data_start_ts,
        "data_end_ts": data_end_ts,
        "data_start_close": entry_bh,
        "data_end_close": exit_bh,
    }

    return summary, tdf, edf, df


def save_backtest_outputs(root: Path, out_dir: str, summary: dict, trades: pd.DataFrame, equity: pd.DataFrame):
    out = (root / out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    trade_cols = ["entry_ts","exit_ts","entry","exit","qty","gross","fees","net","reason"]
    eq_cols = ["ts","equity","drawdown","position_qty"]

    trades_safe = trades.reindex(columns=trade_cols) if isinstance(trades, pd.DataFrame) else pd.DataFrame(columns=trade_cols)
    equity_safe = equity.reindex(columns=eq_cols) if isinstance(equity, pd.DataFrame) else pd.DataFrame(columns=eq_cols)

    trades_safe.to_csv(out / "trades.csv", index=False)
    equity_safe.to_csv(out / "equity.csv", index=False)
