from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src_v2.data_loader import LoadSpec, load_ohlcv
from src_v2.execution import simulate_v2
from src.labels import fixed_horizon_label, triple_barrier_label
from src_v2pro.features import add_features, feature_columns
from src_v2pro.models import train_calibrated_model, choose_threshold_from_cal
from src_v2pro.signals import primary_setups_pro
from src_v2pro.validation import walkforward_windows, apply_purge_embargo


def _interval_minutes(interval: str) -> int:
    return int(interval[:-1]) * {"m": 1, "h": 60, "d": 1440}[interval[-1]]


def run_backtest_v2pro(root: Path, symbol: str, interval: str, date_from: str, date_to: str, cash: float):
    cfg = yaml.safe_load((root / "config_v2_pro.yaml").read_text())

    spec = LoadSpec(symbol=symbol, interval=interval, date_from=date_from, date_to=date_to, market_type=cfg.get("market", {}).get("type", "spot"))
    df, source = load_ohlcv(root, spec)
    df = add_features(df)
    df = pd.concat([df, primary_setups_pro(df)], axis=1)

    horizon_bars = int(cfg["validation"].get("embargo_bars", 24))
    df["label_tb"] = triple_barrier_label(df, horizon=horizon_bars, up_mult=1.5, dn_mult=1.0)
    df["label_fh"] = fixed_horizon_label(df, horizon=horizon_bars, threshold=0.0)
    df["label"] = df["label_tb"].fillna(df["label_fh"])

    feats = feature_columns(df)
    ml = df.dropna(subset=feats + ["label", "setup_any"]).copy()
    ml = ml[ml["setup_any"] == 1].copy()

    vcfg = cfg["validation"]
    min_tr = int(vcfg.get("min_train_events", 300))
    min_cal = int(vcfg.get("min_cal_events", 80))
    min_te = int(vcfg.get("min_test_events", 60))

    step_minutes = _interval_minutes(interval)
    stitched_trades, stitched_eq = [], []
    windows = []

    for tr_start, tr_end, cal_end, te_end in walkforward_windows(
        date_from, date_to,
        int(vcfg["train_days"]), int(vcfg["cal_days"]), int(vcfg["test_days"]), int(vcfg["step_days"]),
    ):
        tr_mask = (ml.ts >= tr_start) & (ml.ts < tr_end)
        cal_mask = (ml.ts >= tr_end) & (ml.ts < cal_end)
        te_mask = (ml.ts >= cal_end) & (ml.ts < te_end)

        tr_mask, cal_mask = apply_purge_embargo(ml, tr_mask, cal_mask, cal_end, te_end, horizon_bars, step_minutes)

        tr = ml[tr_mask].copy(); cal = ml[cal_mask].copy(); te = ml[te_mask].copy()
        if len(tr) < min_tr or len(cal) < min_cal or len(te) < min_te:
            continue
        if tr.label.nunique() < 2 or cal.label.nunique() < 2:
            continue

        model = train_calibrated_model(tr[feats], tr.label.astype(int), cal[feats], cal.label.astype(int))
        p_cal = model.predict_proba(cal[feats])[:, 1]
        thr = choose_threshold_from_cal(
            p_cal,
            cal.label.astype(int).values,
            min_precision=float(cfg["model"].get("min_precision_cal", 0.53)),
            floor=float(cfg["model"].get("threshold_floor", 0.55)),
            cap=float(cfg["model"].get("threshold_cap", 0.90)),
        )

        p_te = pd.Series(model.predict_proba(te[feats])[:, 1], index=te.index)
        seg = df[(df.ts >= cal_end) & (df.ts < te_end)].copy()
        seg["meta_proba"] = seg.index.map(p_te.to_dict()).fillna(0.0)
        seg["trade_flag"] = ((seg["setup_any"] == 1) & (seg["meta_proba"] >= thr)).astype(int)

        # Ensure minimum activity; if too strict, relax threshold to calibration 60th percentile
        min_trades_window = int(cfg["model"].get("min_trades_per_window", 3))
        if int(seg["trade_flag"].sum()) < min_trades_window:
            relaxed_thr = float(pd.Series(p_cal).quantile(0.60))
            thr = min(thr, max(relaxed_thr, float(cfg["model"].get("threshold_floor", 0.52))))
            seg["trade_flag"] = ((seg["setup_any"] == 1) & (seg["meta_proba"] >= thr)).astype(int)

        # map pro config -> simulate_v2 expected keys
        cfg_exec = {
            "exchange": {
                "taker_fee": float(cfg["execution"].get("taker_fee", 0.001)),
                "maker_fee": float(cfg["execution"].get("maker_fee", 0.0005)),
                "slippage_bps": float(cfg["execution"].get("slippage_bps_base", 2.0)),
            },
            "risk": {
                "risk_per_trade": float(cfg["risk"].get("risk_per_trade", 0.004)),
                "risk_per_trade_max": float(cfg["risk"].get("risk_per_trade_max", 0.012)),
                "max_position_pct": float(cfg["risk"].get("max_position_pct", 0.20)),
                "min_notional": float(cfg["risk"].get("min_notional", 10.0)),
                "min_stop_distance_bps": 15,
                "vol_target_ann": float(cfg["risk"].get("vol_target_ann", 0.16)),
            },
            "strategy": {
                "stop_atr_mult": 1.5,
                "tp_r_multiple": 2.2,
                "max_hold_bars": 64,
            },
        }

        tdf, eq = simulate_v2(seg, cfg_exec, interval, initial_cash=cash)
        if not eq.empty and stitched_eq:
            shift = stitched_eq[-1].iloc[-1]["equity"] - eq.iloc[0]["equity"]
            eq = eq.copy(); eq["equity"] = eq["equity"] + shift

        stitched_trades.append(tdf)
        stitched_eq.append(eq)
        windows.append({
            "train_start": str(tr_start.date()), "train_end": str(tr_end.date()),
            "cal_end": str(cal_end.date()), "test_end": str(te_end.date()),
            "n_train": int(len(tr)), "n_cal": int(len(cal)), "n_test": int(len(te)),
            "threshold": float(thr), "n_trades": int(len(tdf)),
        })

    if not stitched_eq:
        # fallback strict chronological split for low-event configurations
        n = len(ml)
        i1 = int(n * 0.7)
        i2 = int(n * 0.85)
        tr, cal, te = ml.iloc[:i1], ml.iloc[i1:i2], ml.iloc[i2:]
        if len(tr) < 120 or len(cal) < 40 or len(te) < 30 or tr.label.nunique() < 2 or cal.label.nunique() < 2:
            raise RuntimeError("V2PRO: no valid OOS windows")

        model = train_calibrated_model(tr[feats], tr.label.astype(int), cal[feats], cal.label.astype(int))
        p_cal = model.predict_proba(cal[feats])[:, 1]
        thr = choose_threshold_from_cal(
            p_cal,
            cal.label.astype(int).values,
            min_precision=float(cfg["model"].get("min_precision_cal", 0.53)),
            floor=float(cfg["model"].get("threshold_floor", 0.55)),
            cap=float(cfg["model"].get("threshold_cap", 0.90)),
        )
        p_te = pd.Series(model.predict_proba(te[feats])[:, 1], index=te.index)
        te_start = te.ts.min()
        te_end = te.ts.max() + pd.Timedelta(minutes=step_minutes)
        seg = df[(df.ts >= te_start) & (df.ts < te_end)].copy()
        seg["meta_proba"] = seg.index.map(p_te.to_dict()).fillna(0.0)
        seg["trade_flag"] = ((seg["setup_any"] == 1) & (seg["meta_proba"] >= thr)).astype(int)
        min_trades_window = int(cfg["model"].get("min_trades_per_window", 3))
        if int(seg["trade_flag"].sum()) < min_trades_window:
            relaxed_thr = float(pd.Series(p_cal).quantile(0.60))
            thr = min(thr, max(relaxed_thr, float(cfg["model"].get("threshold_floor", 0.52))))
            seg["trade_flag"] = ((seg["setup_any"] == 1) & (seg["meta_proba"] >= thr)).astype(int)

        cfg_exec = {
            "exchange": {
                "taker_fee": float(cfg["execution"].get("taker_fee", 0.001)),
                "maker_fee": float(cfg["execution"].get("maker_fee", 0.0005)),
                "slippage_bps": float(cfg["execution"].get("slippage_bps_base", 2.0)),
            },
            "risk": {
                "risk_per_trade": float(cfg["risk"].get("risk_per_trade", 0.004)),
                "risk_per_trade_max": float(cfg["risk"].get("risk_per_trade_max", 0.012)),
                "max_position_pct": float(cfg["risk"].get("max_position_pct", 0.20)),
                "min_notional": float(cfg["risk"].get("min_notional", 10.0)),
                "min_stop_distance_bps": 15,
                "vol_target_ann": float(cfg["risk"].get("vol_target_ann", 0.16)),
            },
            "strategy": {"stop_atr_mult": 1.5, "tp_r_multiple": 2.2, "max_hold_bars": 64},
        }
        trades, equity = simulate_v2(seg, cfg_exec, interval, initial_cash=cash)
        windows = [{"mode": "fallback_split", "threshold": float(thr), "n_train": int(len(tr)), "n_cal": int(len(cal)), "n_test": int(len(te)), "n_trades": int(len(trades))}]
    else:
        trades = pd.concat(stitched_trades, ignore_index=True)
        equity = pd.concat(stitched_eq, ignore_index=True)

    net = float(trades["net"].sum()) if len(trades) else 0.0
    wins = int((trades["net"] > 0).sum()) if len(trades) else 0
    gp = float(trades.loc[trades.net > 0, "net"].sum()) if len(trades) else 0.0
    gl = float(-trades.loc[trades.net <= 0, "net"].sum()) if len(trades) else 0.0
    pf = gp / gl if gl > 0 else float("inf")

    val_mode = "walk_forward_purged_oos"
    if windows and windows[0].get("mode") == "fallback_split":
        val_mode = "split_oos_fallback"

    summary = {
        "symbol": symbol,
        "interval": interval,
        "date_from": date_from,
        "date_to": date_to,
        "data_source": source,
        "model": "v2pro_calibrated_wf",
        "validation_mode": val_mode,
        "wf_windows": int(len(windows)),
        "windows": windows,
        "trades": int(len(trades)),
        "wins": wins,
        "losses": int(len(trades) - wins),
        "win_rate": float((wins / len(trades)) if len(trades) else 0.0),
        "net_pnl": net,
        "return_pct": float(net / cash),
        "profit_factor": float(pf),
        "max_drawdown": float(equity["drawdown"].max()) if len(equity) else 0.0,
        "fees_paid_total": float(trades["fees"].sum()) if len(trades) and "fees" in trades.columns else 0.0,
    }
    return summary, trades, equity


def save_outputs(root: Path, export_dir: str, summary: dict, trades: pd.DataFrame, equity: pd.DataFrame):
    out = (root / export_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    trades.to_csv(out / "trades.csv", index=False)
    equity.to_csv(out / "equity.csv", index=False)
