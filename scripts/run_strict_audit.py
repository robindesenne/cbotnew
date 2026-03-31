#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import LoadSpec, load_ohlcv
from src.execution import simulate_spot_long_only
from src.features import add_features, feature_columns
from src.labels import fixed_horizon_label, triple_barrier_label
from src.signals import primary_setups


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["15m", "1h", "4h"]
DATE_FROM = "2022-01-01"
DATE_TO = "2026-03-21"
CASH0 = 10_000.0


@dataclass
class WF:
    train_days: int = 180
    test_days: int = 60
    step_days: int = 60
    embargo_bars: int = 24
    horizon_bars: int = 24


def annual_factor(interval: str) -> float:
    m = int(interval[:-1]) * {"m": 1, "h": 60, "d": 1440}[interval[-1]]
    return np.sqrt((365 * 24 * 60) / m)


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, interval: str) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "max_drawdown": float(equity["drawdown"].max()) if not equity.empty else 0.0,
            "turnover": 0.0,
            "exposure": float((equity.get("position_qty", pd.Series(dtype=float)) > 0).mean()) if not equity.empty and "position_qty" in equity else 0.0,
            "sharpe": 0.0,
        }
    gp = float(trades.loc[trades.net > 0, "net"].sum())
    gl = float(-trades.loc[trades.net <= 0, "net"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    net = float(trades.net.sum())
    win = float((trades.net > 0).mean())
    eq = equity.copy()
    rets = eq["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float((rets.mean() / (rets.std() + 1e-12)) * annual_factor(interval)) if len(rets) > 5 else 0.0
    fees_paid_total = float(trades["fees"].sum()) if "fees" in trades.columns and not trades.empty else 0.0
    return {
        "trades": int(len(trades)),
        "win_rate": win,
        "profit_factor": float(pf),
        "net_pnl": net,
        "fees_paid_total": fees_paid_total,
        "return_pct": float(net / CASH0),
        "max_drawdown": float(eq["drawdown"].max()) if "drawdown" in eq else 0.0,
        "turnover": float((trades.entry * trades.qty + trades.exit * trades.qty).sum()),
        "exposure": float((eq["position_qty"] > 0).mean()) if "position_qty" in eq else 0.0,
        "sharpe": sharpe,
    }


def build_dataset(root: Path, symbol: str, interval: str, cfg: dict, wf: WF):
    spec = LoadSpec(symbol=symbol, interval=interval, date_from=DATE_FROM, date_to=DATE_TO, market_type="spot")
    df, source = load_ohlcv(root, spec)
    df = add_features(df)
    setups = primary_setups(df)
    df = pd.concat([df, setups], axis=1)

    df["label_tb"] = triple_barrier_label(df, horizon=wf.horizon_bars, up_mult=1.5, dn_mult=1.0)
    df["label_fh"] = fixed_horizon_label(df, horizon=wf.horizon_bars, threshold=0.0)
    df["label"] = df["label_tb"].fillna(df["label_fh"])

    feats = feature_columns(df)
    ml = df.dropna(subset=feats + ["label", "setup_any"]).copy()
    ml = ml[ml.setup_any == 1].copy()
    ml["event_end_ts"] = ml["ts"].shift(-wf.horizon_bars).ffill()
    return df, ml, feats, source


def model_train_predict(X_train, y_train, X_cal, y_cal, X_test):
    cands = {
        "logreg": make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=10000, solver="saga", C=0.5, class_weight="balanced", random_state=42)),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42),
    }
    best_name, best_auc, best_model = None, -1, None
    for n, m in cands.items():
        m.fit(X_train, y_train)
        p = m.predict_proba(X_cal)[:, 1]
        auc = roc_auc_score(y_cal, p) if len(np.unique(y_cal)) > 1 else 0.5
        if auc > best_auc:
            best_name, best_auc, best_model = n, auc, m

    cal = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    p_cal = cal.predict_proba(X_cal)[:, 1]

    # threshold strictly from calibration split
    best_t, best_f1 = 0.65, -1
    y = y_cal.values
    for t in np.linspace(0.5, 0.9, 41):
        pred = (p_cal >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        if prec < 0.52:
            continue
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t

    p_test = cal.predict_proba(X_test)[:, 1]
    return best_name, float(best_auc), float(best_t), p_test, cal


def run_purged_wf(root: Path, symbol: str, interval: str, cfg: dict, wf: WF, stress: dict | None = None, placebo: str | None = None):
    df, ml, feats, source = build_dataset(root, symbol, interval, cfg, wf)
    if len(ml) < 250:
        return None

    d0 = pd.Timestamp(DATE_FROM, tz="UTC")
    d1 = pd.Timestamp(DATE_TO, tz="UTC")

    cash = CASH0
    stitched_trades = []
    stitched_eq = []
    windows = []
    top_features = []

    cur = d0
    while cur + pd.Timedelta(days=wf.train_days + wf.test_days) <= d1:
        tr_start = cur
        tr_end = cur + pd.Timedelta(days=wf.train_days)
        te_end = tr_end + pd.Timedelta(days=wf.test_days)

        test_mask = (ml.ts >= tr_end) & (ml.ts < te_end)
        train_mask = (ml.ts >= tr_start) & (ml.ts < tr_end)

        # purge overlaps around labels + embargo
        embargo = pd.Timedelta(minutes=wf.embargo_bars * (int(interval[:-1]) * {"m": 1, "h": 60, "d": 1440}[interval[-1]]))
        te_start_emb = tr_end - embargo
        te_end_emb = te_end + embargo
        overlap = (ml.event_end_ts >= te_start_emb) & (ml.ts <= te_end_emb)
        train_mask = train_mask & (~overlap)

        train = ml[train_mask].copy(); test = ml[test_mask].copy()
        if len(train) < 180 or len(test) < 40:
            cur += pd.Timedelta(days=wf.step_days)
            continue

        # calibration split strictly within train
        split = int(len(train) * 0.8)
        tr2 = train.iloc[:split]; cal = train.iloc[split:]
        if len(tr2) < 120 or len(cal) < 40:
            cur += pd.Timedelta(days=wf.step_days)
            continue

        ytr = tr2.label.astype(int)
        ycal = cal.label.astype(int)
        yte = test.label.astype(int)

        if placebo == "random_label":
            ytr = ytr.sample(frac=1.0, random_state=42).reset_index(drop=True)
            ytr.index = tr2.index
        Xtr = tr2[feats].copy(); Xcal = cal[feats].copy(); Xte = test[feats].copy()
        if placebo == "permute_features":
            rs = np.random.RandomState(42)
            for c in feats:
                Xtr[c] = rs.permutation(Xtr[c].values)
                Xcal[c] = rs.permutation(Xcal[c].values)
                Xte[c] = rs.permutation(Xte[c].values)

        name, auc, thr, pte, model = model_train_predict(Xtr, ytr, Xcal, ycal, Xte)

        # feature importance top20
        imp = {}
        if hasattr(model.estimator, "feature_importances_"):
            vals = model.estimator.feature_importances_
            imp = dict(sorted(zip(feats, vals), key=lambda x: x[1], reverse=True)[:20])
        elif hasattr(model.estimator, "coef_"):
            vals = np.abs(model.estimator.coef_[0])
            imp = dict(sorted(zip(feats, vals), key=lambda x: x[1], reverse=True)[:20])
        top_features.append({"window_start": str(tr_start.date()), "top20": imp})

        # map proba only on test timestamps
        seg = df[(df.ts >= tr_end) & (df.ts < te_end)].copy()
        seg["meta_proba"] = 0.0
        idx_map = dict(zip(test.ts.astype(str), pte))
        seg["meta_proba"] = seg.ts.astype(str).map(idx_map).fillna(0.0)
        seg["trade_flag"] = ((seg.setup_any == 1) & (seg.meta_proba >= thr)).astype(int)

        cfg_run = copy.deepcopy(cfg)
        if stress:
            cfg_run["exchange"]["taker_fee"] *= stress.get("fee_mult", 1.0)
            cfg_run["exchange"]["maker_fee"] *= stress.get("fee_mult", 1.0)
            cfg_run["exchange"]["slippage_bps"] *= stress.get("slippage_mult", 1.0)
        if stress and stress.get("vol_slippage", False):
            # conservative proxy: inflate slippage based on realized vol percentile on segment
            base = cfg_run["exchange"]["slippage_bps"]
            vol_boost = float(seg["rv_24"].rank(pct=True).fillna(0.5).mean())
            cfg_run["exchange"]["slippage_bps"] = base * (1.0 + vol_boost)

        tdf, eq = simulate_spot_long_only(seg, cfg_run)
        if not eq.empty:
            # stitch with cash carry approximation
            if stitched_eq:
                shift = stitched_eq[-1]["equity"] - eq.iloc[0]["equity"]
                eq = eq.copy(); eq["equity"] = eq["equity"] + shift
            cash = float(eq.iloc[-1]["equity"])

        stitched_trades.append(tdf)
        stitched_eq.append(eq)
        windows.append({"train_start": str(tr_start.date()), "train_end": str(tr_end.date()), "test_end": str(te_end.date()), "model": name, "auc_cal": auc, "threshold": thr, "n_test_events": int(len(test))})

        cur += pd.Timedelta(days=wf.step_days)

    if not stitched_eq:
        return None

    trades = pd.concat(stitched_trades, ignore_index=True) if stitched_trades else pd.DataFrame()
    equity = pd.concat(stitched_eq, ignore_index=True) if stitched_eq else pd.DataFrame()
    met = compute_metrics(trades, equity, interval)

    # diagnostics
    if not trades.empty:
        t = trades.copy(); t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True)
        by_year = t.groupby(t.exit_ts.dt.year)["net"].sum().to_dict()
        by_q = t.groupby(t.exit_ts.dt.to_period("Q").astype(str))["net"].sum().to_dict()
    else:
        by_year, by_q = {}, {}

    out = {
        "symbol": symbol,
        "interval": interval,
        "data_source": source,
        "mode": "purged_wf_oos_stitched",
        "stress": stress or {},
        "placebo": placebo,
        **met,
        "windows": windows,
        "pnl_by_year": by_year,
        "pnl_by_quarter": by_q,
        "top_features_stability": top_features,
    }
    return out, trades, equity





def run_primary_only_oos(root: Path, symbol: str, interval: str, cfg: dict, wf: WF):
    df, ml, _, source = build_dataset(root, symbol, interval, cfg, wf)
    if len(df) < 300:
        return None

    d0 = pd.Timestamp(DATE_FROM, tz="UTC")
    d1 = pd.Timestamp(DATE_TO, tz="UTC")
    stitched_trades=[]; stitched_eq=[]
    cur=d0
    while cur + pd.Timedelta(days=wf.train_days + wf.test_days) <= d1:
        tr_end = cur + pd.Timedelta(days=wf.train_days)
        te_end = tr_end + pd.Timedelta(days=wf.test_days)
        seg = df[(df.ts >= tr_end) & (df.ts < te_end)].copy()
        if len(seg) < 80:
            cur += pd.Timedelta(days=wf.step_days); continue
        seg['trade_flag'] = seg['setup_any'].astype(int)
        tdf, eq = simulate_spot_long_only(seg, cfg)
        if not eq.empty and stitched_eq:
            shift = stitched_eq[-1]['equity'] - eq.iloc[0]['equity']
            eq=eq.copy(); eq['equity']=eq['equity']+shift
        stitched_trades.append(tdf); stitched_eq.append(eq)
        cur += pd.Timedelta(days=wf.step_days)

    if not stitched_eq:
        return None
    trades = pd.concat(stitched_trades, ignore_index=True)
    equity = pd.concat(stitched_eq, ignore_index=True)
    met=compute_metrics(trades, equity, interval)
    return {'symbol':symbol,'interval':interval,'mode':'primary_only_oos',**met}, trades, equity

def compute_verdict(base_runs: pd.DataFrame) -> str:
    pf_med = base_runs.profit_factor.median()
    dd_med = base_runs.max_drawdown.median()
    pass_count = int((base_runs.profit_factor > 1.10).sum())
    if pf_med > 1.15 and dd_med < 0.20 and pass_count >= max(1, int(len(base_runs) * 0.6)):
        return "prometteur mais fragile"
    if pf_med > 1.25 and dd_med < 0.15 and pass_count >= int(len(base_runs) * 0.75):
        return "robuste"
    return "invalide / probablement biaisé"


def _k(sym: str, tf: str) -> str:
    return f"{sym}_{tf}".replace('/', '_')


def main():
    root = ROOT
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    wf = WF()

    out_dir = root / "reports" / "strict_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_rows = []
    primary_rows = []
    stress_rows = []
    placebo_rows = []

    stress_tests = [
        {"name": "base", "fee_mult": 1.0, "slippage_mult": 1.0, "vol_slippage": False},
        {"name": "fee_x1_5", "fee_mult": 1.5, "slippage_mult": 1.0, "vol_slippage": False},
        {"name": "fee_x2", "fee_mult": 2.0, "slippage_mult": 1.0, "vol_slippage": False},
        {"name": "slip_x2", "fee_mult": 1.0, "slippage_mult": 2.0, "vol_slippage": False},
        {"name": "vol_dependent_slip", "fee_mult": 1.0, "slippage_mult": 1.0, "vol_slippage": True},
    ]

    for sym in SYMBOLS:
        for tf in INTERVALS:
            key = _k(sym, tf)
            base_json = runs_dir / f"base_{key}.json"
            stress_json = runs_dir / f"stress_{key}.json"
            placebo_json = runs_dir / f"placebo_{key}.json"

            if base_json.exists():
                summary = json.loads(base_json.read_text())
                base_rows.append(summary)
            else:
                try:
                    res = run_purged_wf(root, sym, tf, cfg, wf)
                    if res is None:
                        continue
                    summary, trades, equity = res
                    (out_dir / f"trades_oos_{sym}_{tf}.csv").write_text(trades.to_csv(index=False))
                    (out_dir / f"equity_oos_{sym}_{tf}.csv").write_text(equity.to_csv(index=False))
                    base_rows.append(summary)
                    base_json.write_text(json.dumps(summary, indent=2))
                except Exception as e:
                    (runs_dir / f"error_base_{key}.txt").write_text(str(e))
                    continue

            # primary-only stitched OOS baseline
            p_json = runs_dir / f"primary_{key}.json"
            if p_json.exists():
                primary_rows.append(json.loads(p_json.read_text()))
            else:
                try:
                    pr = run_primary_only_oos(root, sym, tf, cfg, wf)
                    if pr is not None:
                        ps, _, _ = pr
                        primary_rows.append(ps)
                        p_json.write_text(json.dumps(ps, indent=2))
                except Exception as e:
                    (runs_dir / f"error_primary_{key}.txt").write_text(str(e))

            if stress_json.exists():
                stress_rows.extend(json.loads(stress_json.read_text()))
            else:
                srows = []
                for st in stress_tests[1:]:
                    try:
                        r = run_purged_wf(root, sym, tf, cfg, wf, stress=st)
                        if r is None:
                            continue
                        s2, _, _ = r
                        srows.append({"symbol": sym, "interval": tf, "stress": st["name"], "profit_factor": s2["profit_factor"], "return_pct": s2["return_pct"], "max_drawdown": s2["max_drawdown"]})
                    except Exception as e:
                        (runs_dir / f"error_stress_{key}_{st['name']}.txt").write_text(str(e))
                stress_rows.extend(srows)
                stress_json.write_text(json.dumps(srows, indent=2))

            if placebo_json.exists():
                placebo_rows.extend(json.loads(placebo_json.read_text()))
            else:
                prows = []
                for pb in ["random_label", "permute_features"]:
                    try:
                        r = run_purged_wf(root, sym, tf, cfg, wf, placebo=pb)
                        if r is None:
                            continue
                        s2, _, _ = r
                        prows.append({"symbol": sym, "interval": tf, "placebo": pb, "profit_factor": s2["profit_factor"], "return_pct": s2["return_pct"]})
                    except Exception as e:
                        (runs_dir / f"error_placebo_{key}_{pb}.txt").write_text(str(e))
                placebo_rows.extend(prows)
                placebo_json.write_text(json.dumps(prows, indent=2))

    base_df = pd.DataFrame(base_rows)
    primary_df = pd.DataFrame(primary_rows)
    stress_df = pd.DataFrame(stress_rows)
    placebo_df = pd.DataFrame(placebo_rows)

    base_df.to_csv(out_dir / "strict_oos_summary.csv", index=False)
    primary_df.to_csv(out_dir / "primary_only_oos_summary.csv", index=False)
    stress_df.to_csv(out_dir / "stress_sensitivity.csv", index=False)
    placebo_df.to_csv(out_dir / "placebo_tests.csv", index=False)

    verdict = compute_verdict(base_df) if not base_df.empty else "invalide / probablement biaisé"

    # audit report
    report = {
        "anti_lookahead": {
            "htf_features_present": False,
            "note": "Current strict pipeline uses same-timeframe features only; no HTF merge/lookahead path used.",
            "label_contamination_control": "Purged walk-forward with overlap purge + embargo implemented.",
        },
        "calibration_threshold": {
            "calibration_fit_scope": "train-calibration split only",
            "threshold_scope": "chosen on calibration split only (never test)",
        },
        "verdict": verdict,
        "go_no_go_paper_trading": "NO-GO" if verdict.startswith("invalide") else "GO_PAPER_ONLY",
    }

    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2))

    def _md_table(df):
        if df.empty:
            return "No data"
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)

    md = [
        "# Strict Validation Audit",
        "",
        "## Verdict",
        f"- **{verdict}**",
        f"- Decision: **{report['go_no_go_paper_trading']}**",
        "",
        "## Base purged OOS summary",
        _md_table(base_df),
        "",
        "## Primary-only OOS baseline",
        _md_table(primary_df),
        "",
        "## Stress sensitivity",
        _md_table(stress_df),
        "",
        "## Placebo tests",
        _md_table(placebo_df),
    ]
    (out_dir / "audit_report.md").write_text("\n".join(md))

    print("Strict audit completed")
    if not base_df.empty:
        cols = [c for c in ["symbol", "interval", "profit_factor", "max_drawdown", "return_pct", "trades"] if c in base_df.columns]
        print(base_df[cols].to_string(index=False))
    else:
        print("No base rows generated")
    print("Verdict:", verdict)


if __name__ == "__main__":
    main()
