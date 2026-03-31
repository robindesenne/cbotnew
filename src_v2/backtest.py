from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src_v2.data_loader import LoadSpec, load_ohlcv
from src_v2.execution import simulate_v2
from src_v2.features import add_features, feature_columns
from src.labels import fixed_horizon_label, triple_barrier_label
from src_v2.models import train_meta_model
from src_v2.regime import detect_regime_v2, regime_threshold_adjustment
from src_v2.signals import dynamic_threshold, primary_setups_v2


def _interval_minutes(interval: str) -> int:
    return int(interval[:-1]) * {"m": 1, "h": 60, "d": 1440}[interval[-1]]


def _compute_summary(symbol: str, interval: str, date_from: str, date_to: str, source: str, df: pd.DataFrame, tdf: pd.DataFrame, eq: pd.DataFrame, cash: float, cfg: dict, extra: dict | None = None) -> dict:
    net_pnl = float(tdf["net"].sum()) if len(tdf) else 0.0
    fees = float(tdf["fees"].sum()) if len(tdf) and "fees" in tdf.columns else 0.0
    gp = float(tdf.loc[tdf.net > 0, "net"].sum()) if len(tdf) else 0.0
    gl = float(-tdf.loc[tdf.net <= 0, "net"].sum()) if len(tdf) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    max_dd = float(eq["drawdown"].max()) if len(eq) else 0.0
    wins = int((tdf.net > 0).sum()) if len(tdf) else 0

    if len(df) >= 2:
        e0 = float(df.iloc[0].close)
        e1 = float(df.iloc[-1].close)
        ex = cfg.get("exchange", {})
        bh_cost = float(ex.get("taker_fee", 0.001)) * 2 + (float(ex.get("slippage_bps", 2.0)) / 10000.0) * 2
        bh = (e1 / e0 - 1.0) - bh_cost
    else:
        bh = 0.0

    data_start_ts = str(df.iloc[0].ts) if len(df) else None
    data_end_ts = str(df.iloc[-1].ts) if len(df) else None

    out = {
        "symbol": symbol,
        "interval": interval,
        "date_from": date_from,
        "date_to": date_to,
        "data_source": source,
        "model": "v2_wf_oos",
        "trades": int(len(tdf)),
        "wins": wins,
        "losses": int(len(tdf) - wins),
        "win_rate": float((wins / len(tdf)) if len(tdf) else 0.0),
        "net_pnl": net_pnl,
        "return_pct": float(net_pnl / cash),
        "profit_factor": float(pf),
        "max_drawdown": max_dd,
        "fees_paid_total": fees,
        "bh_return_pct": float(bh),
        "alpha_vs_bh_pct": float((net_pnl / cash) - bh),
        "data_start_ts": data_start_ts,
        "data_end_ts": data_end_ts,
        "warning": "no_trades" if int(len(tdf)) == 0 else None,
    }
    if extra:
        out.update(extra)
    return out


def run_backtest_v2(root: Path, symbol: str, interval: str, date_from: str, date_to: str, cash: float):
    cfg = yaml.safe_load((root / "config_v2.yaml").read_text())
    # Backward compatibility: map v2 namespaces to execution config expected by simulator
    if "risk" not in cfg:
        cfg["risk"] = {}
    if "risk_v2" in cfg and isinstance(cfg["risk_v2"], dict):
        for k, v in cfg["risk_v2"].items():
            if k not in cfg["risk"]:
                cfg["risk"][k] = v

    spec = LoadSpec(symbol=symbol, interval=interval, date_from=date_from, date_to=date_to, market_type=cfg.get("market", {}).get("type", "spot"))
    df, source = load_ohlcv(root, spec)
    df = add_features(df)
    setups = primary_setups_v2(df)
    df = pd.concat([df, setups], axis=1)
    df["regime"] = detect_regime_v2(df)

    horizon_bars = 24
    df["label_tb"] = triple_barrier_label(df, horizon=horizon_bars, up_mult=1.5, dn_mult=1.0)
    df["label_fh"] = fixed_horizon_label(df, horizon=horizon_bars, threshold=0.0)
    df["label"] = df["label_tb"].fillna(df["label_fh"])

    feats = feature_columns(df)
    ml = df.dropna(subset=feats + ["label", "setup_any"]).copy()
    ml = ml[ml.setup_any == 1].copy()
    if len(ml) < 500:
        raise RuntimeError("V2: not enough events")

    bar_min = _interval_minutes(interval)
    embargo = pd.Timedelta(minutes=bar_min * horizon_bars)

    train_days = int(cfg.get("validation", {}).get("wf_train_days", 180))
    cal_days = int(cfg.get("validation", {}).get("wf_cal_days", 60))
    test_days = int(cfg.get("validation", {}).get("wf_test_days", 60))
    step_days = int(cfg.get("validation", {}).get("wf_step_days", 60))

    d0 = pd.Timestamp(date_from, tz="UTC")
    d1 = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)

    stitched_trades: list[pd.DataFrame] = []
    stitched_eq: list[pd.DataFrame] = []
    windows = []

    cur = d0
    while cur + pd.Timedelta(days=train_days + cal_days + test_days) <= d1:
        tr_start = cur
        tr_end = tr_start + pd.Timedelta(days=train_days)
        cal_end = tr_end + pd.Timedelta(days=cal_days)
        te_end = cal_end + pd.Timedelta(days=test_days)

        tr_mask = (ml.ts >= tr_start) & (ml.ts < tr_end)
        cal_mask = (ml.ts >= tr_end) & (ml.ts < cal_end)
        te_mask = (ml.ts >= cal_end) & (ml.ts < te_end)

        # purge overlap around test labels
        ml["event_end_ts"] = ml["ts"].shift(-horizon_bars).ffill()
        te_start_emb = cal_end - embargo
        te_end_emb = te_end + embargo
        overlap = (ml.event_end_ts >= te_start_emb) & (ml.ts <= te_end_emb)
        tr_mask = tr_mask & (~overlap)
        cal_mask = cal_mask & (~overlap)

        tr = ml[tr_mask].copy()
        cal = ml[cal_mask].copy()
        te = ml[te_mask].copy()

        if len(tr) < 180 or len(cal) < 60 or len(te) < 40:
            cur += pd.Timedelta(days=step_days)
            continue
        if tr.label.nunique() < 2 or cal.label.nunique() < 2:
            cur += pd.Timedelta(days=step_days)
            continue

        name, model = train_meta_model(tr[feats], tr.label.astype(int), cal[feats], cal.label.astype(int))

        p_cal = pd.Series(model.predict_proba(cal[feats])[:, 1], index=cal.index)
        base_thr = dynamic_threshold(p_cal)

        p_te = pd.Series(model.predict_proba(te[feats])[:, 1], index=te.index)

        seg = df[(df.ts >= cal_end) & (df.ts < te_end)].copy()
        seg["meta_proba"] = seg.index.map(p_te.to_dict()).fillna(0.0)
        seg["thr_dyn"] = base_thr
        seg["thr_dyn"] = seg.apply(lambda r: min(0.95, max(0.5, base_thr + regime_threshold_adjustment(str(r.get("regime", "neutral"))))), axis=1)

        trade_flag = (seg["setup_any"] == 1) & (seg["meta_proba"] >= seg["thr_dyn"]) & (~seg["regime"].isin(["bear", "chop"]))
        if int(trade_flag.sum()) == 0:
            trade_flag = (seg["setup_any"] == 1) & (seg["meta_proba"] >= (base_thr + 0.05)) & (~seg["regime"].isin(["bear"]))
        seg["trade_flag"] = trade_flag.astype(int)

        tdf, eq = simulate_v2(seg, cfg, interval, initial_cash=cash)
        if not eq.empty and stitched_eq:
            shift = stitched_eq[-1].iloc[-1]["equity"] - eq.iloc[0]["equity"]
            eq = eq.copy()
            eq["equity"] = eq["equity"] + shift

        stitched_trades.append(tdf)
        stitched_eq.append(eq)
        windows.append({
            "train_start": str(tr_start.date()),
            "train_end": str(tr_end.date()),
            "cal_end": str(cal_end.date()),
            "test_end": str(te_end.date()),
            "model": name,
            "threshold_base": float(base_thr),
            "n_train": int(len(tr)),
            "n_cal": int(len(cal)),
            "n_test": int(len(te)),
            "n_trades": int(len(tdf)),
        })

        cur += pd.Timedelta(days=step_days)

    if not stitched_eq:
        # Fallback for short ranges: strict chronological split, trade test segment only
        n = len(ml)
        i1 = int(n * 0.7)
        i2 = int(n * 0.85)
        tr, cal, te = ml.iloc[:i1], ml.iloc[i1:i2], ml.iloc[i2:]
        if len(tr) < 180 or len(cal) < 60 or len(te) < 40 or tr.label.nunique() < 2 or cal.label.nunique() < 2:
            raise RuntimeError("V2 WF: no valid windows generated")

        name, model = train_meta_model(tr[feats], tr.label.astype(int), cal[feats], cal.label.astype(int))
        p_cal = pd.Series(model.predict_proba(cal[feats])[:, 1], index=cal.index)
        base_thr = dynamic_threshold(p_cal)
        p_te = pd.Series(model.predict_proba(te[feats])[:, 1], index=te.index)

        te_start = te.ts.min()
        te_end = te.ts.max() + pd.Timedelta(minutes=bar_min)
        seg = df[(df.ts >= te_start) & (df.ts < te_end)].copy()
        seg["meta_proba"] = seg.index.map(p_te.to_dict()).fillna(0.0)
        seg["thr_dyn"] = seg.apply(lambda r: min(0.95, max(0.5, base_thr + regime_threshold_adjustment(str(r.get("regime", "neutral"))))), axis=1)
        trade_flag = (seg["setup_any"] == 1) & (seg["meta_proba"] >= seg["thr_dyn"]) & (~seg["regime"].isin(["bear", "chop"]))
        if int(trade_flag.sum()) == 0:
            trade_flag = (seg["setup_any"] == 1) & (seg["meta_proba"] >= (base_thr + 0.05)) & (~seg["regime"].isin(["bear"]))
        seg["trade_flag"] = trade_flag.astype(int)

        tdf, eq = simulate_v2(seg, cfg, interval, initial_cash=cash)
        trades = tdf.copy()
        equity = eq.copy()
        windows = [{"mode": "fallback_split", "model": name, "threshold_base": float(base_thr), "n_train": int(len(tr)), "n_cal": int(len(cal)), "n_test": int(len(te)), "n_trades": int(len(tdf))}]
    else:
        trades = pd.concat(stitched_trades, ignore_index=True) if stitched_trades else pd.DataFrame()
        equity = pd.concat(stitched_eq, ignore_index=True) if stitched_eq else pd.DataFrame()

    mode = "walk_forward_purged_oos"
    if windows and windows[0].get("mode") == "fallback_split":
        mode = "split_oos_fallback"

    summary = _compute_summary(
        symbol=symbol,
        interval=interval,
        date_from=date_from,
        date_to=date_to,
        source=source,
        df=df,
        tdf=trades,
        eq=equity,
        cash=cash,
        cfg=cfg,
        extra={
            "windows": windows,
            "wf_windows": int(len(windows)),
            "validation_mode": mode,
        },
    )
    return summary, trades, equity


def save_outputs(root: Path, export_dir: str, summary: dict, trades: pd.DataFrame, equity: pd.DataFrame):
    out = (root / export_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    trade_cols = ["entry_ts", "exit_ts", "entry", "exit", "qty", "gross", "fees", "net", "reason"]
    eq_cols = ["ts", "equity", "drawdown", "position_qty"]

    trades_safe = trades.reindex(columns=trade_cols) if isinstance(trades, pd.DataFrame) else pd.DataFrame(columns=trade_cols)
    equity_safe = equity.reindex(columns=eq_cols) if isinstance(equity, pd.DataFrame) else pd.DataFrame(columns=eq_cols)

    trades_safe.to_csv(out / "trades.csv", index=False)
    equity_safe.to_csv(out / "equity.csv", index=False)
