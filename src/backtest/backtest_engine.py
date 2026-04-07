from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data_loader import LoadSpec, load_ohlcv
from ..execution import simulate_spot_long_only
from ..features import add_features, feature_columns
from ..labels import fixed_horizon_label, triple_barrier_label
from ..regime import regime_rules, regime_transition_prob
from ..signals import final_trade_flag, primary_setups
from .config import GLOBAL_CONFIG
from .solusdt_pipeline import SolusdtPipelineSpec, build_solusdt_dataset


@dataclass
class Candidate:
    name: str
    model: object
    params: Dict


def _warn_leakage(df: pd.DataFrame) -> List[str]:
    warnings = []
    if "ts" in df.columns and not df["ts"].is_monotonic_increasing:
        warnings.append("LEAKAGE_WARNING: timestamps not monotonic increasing")
    for col in df.columns:
        if "future" in col.lower() or "target" in col.lower():
            warnings.append(f"LEAKAGE_WARNING: suspicious column name '{col}'")
    return warnings


def _chrono_slices(n: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # expanding train, rolling test
    chunk = n // (n_splits + 1)
    out = []
    for i in range(1, n_splits + 1):
        tr_end = chunk * i
        te_end = chunk * (i + 1) if i < n_splits else n
        tr_idx = np.arange(0, tr_end)
        te_idx = np.arange(tr_end, te_end)
        if len(tr_idx) > 0 and len(te_idx) > 0:
            out.append((tr_idx, te_idx))
    return out


def _purge_embargo_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    event_end_idx: np.ndarray,
    embargo_bars: int,
) -> np.ndarray:
    test_start = int(test_idx.min())
    test_end = int(test_idx.max())
    banned = np.zeros(event_end_idx.shape[0], dtype=bool)
    for i in test_idx:
        ee = int(event_end_idx[i]) if int(event_end_idx[i]) >= i else int(i)
        left = i
        right = ee + embargo_bars
        banned[left : min(right + 1, len(banned))] = True

    safe_train = []
    for i in train_idx:
        ee = int(event_end_idx[i]) if int(event_end_idx[i]) >= i else int(i)
        overlap = (i <= test_end + embargo_bars) and (ee >= test_start)
        if overlap:
            continue
        if banned[i]:
            continue
        safe_train.append(i)
    return np.array(safe_train, dtype=int)


def _candidates() -> List[Candidate]:
    return [
        Candidate(
            "logreg_c05",
            make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=10000, solver="saga", C=0.5, class_weight="balanced", random_state=42),
            ),
            {"family": "logreg", "C": 0.5},
        ),
        Candidate(
            "logreg_c10",
            make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=10000, solver="saga", C=1.0, class_weight="balanced", random_state=42),
            ),
            {"family": "logreg", "C": 1.0},
        ),
        Candidate(
            "rf_d8",
            RandomForestClassifier(n_estimators=350, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1),
            {"family": "rf", "depth": 8},
        ),
        Candidate(
            "hgb_d6",
            HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=450, random_state=42),
            {"family": "hgb", "depth": 6},
        ),
    ]


def _best_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    best_t, best_score = 0.5, -1.0
    for t in GLOBAL_CONFIG["threshold_grid"]:
        pred = (proba >= t).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        if prec < GLOBAL_CONFIG["min_precision"]:
            continue
        tp = ((pred == 1) & (y_true.values == 1)).sum()
        fp = ((pred == 1) & (y_true.values == 0)).sum()
        fn = ((pred == 0) & (y_true.values == 1)).sum()
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_score:
            best_score, best_t = f1, t
    return float(best_t)


def _nested_select(
    X: pd.DataFrame,
    y: pd.Series,
    event_end_idx: np.ndarray,
) -> Tuple[object, str, float, List[Dict]]:
    inner_logs: List[Dict] = []
    splits = _chrono_slices(len(X), GLOBAL_CONFIG["inner_splits"])
    model_scores: Dict[str, List[float]] = {}
    threshold_bank: Dict[str, List[float]] = {}

    for c in _candidates():
        model_scores[c.name] = []
        threshold_bank[c.name] = []

    for k, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_idx_safe = _purge_embargo_train_indices(
            tr_idx, va_idx, event_end_idx=event_end_idx, embargo_bars=GLOBAL_CONFIG["embargo_bars"]
        )
        if len(tr_idx_safe) < GLOBAL_CONFIG["min_train_rows"] or len(va_idx) < GLOBAL_CONFIG["min_valid_rows"]:
            continue

        Xtr, ytr = X.iloc[tr_idx_safe], y.iloc[tr_idx_safe]
        Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
        if ytr.nunique() < 2 or yva.nunique() < 2:
            continue

        for cand in _candidates():
            m = cand.model
            m.fit(Xtr, ytr)
            pva = m.predict_proba(Xva)[:, 1]
            auc = roc_auc_score(yva, pva)
            thr = _best_threshold(yva, pva)
            model_scores[cand.name].append(float(auc))
            threshold_bank[cand.name].append(float(thr))
            inner_logs.append({"fold": k, "model": cand.name, "auc_val": float(auc), "threshold": float(thr)})

    ranked = []
    for name, aucs in model_scores.items():
        if not aucs:
            continue
        ranked.append((name, float(np.mean(aucs))))
    if not ranked:
        fallback = _candidates()[0]
        fallback.model.fit(X, y)
        return fallback.model, fallback.name, 0.5, inner_logs

    best_name = sorted(ranked, key=lambda x: x[1], reverse=True)[0][0]
    best_cand = [c for c in _candidates() if c.name == best_name][0]

    final_model = best_cand.model
    final_model.fit(X, y)
    chosen_threshold = float(np.median(threshold_bank.get(best_name, [0.5])))
    return final_model, best_name, chosen_threshold, inner_logs


def _oos_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> Dict[str, float]:
    if equity is None or equity.empty:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
        }

    rets = equity["equity"].pct_change().dropna()
    if len(rets) > 2:
        mu = float(rets.mean())
        sigma = float(rets.std() + 1e-12)
        neg_sigma = float(rets[rets < 0].std() + 1e-12)
        sharpe = mu / sigma * np.sqrt(365 * 24 * 4)
        sortino = mu / neg_sigma * np.sqrt(365 * 24 * 4)
    else:
        sharpe = 0.0
        sortino = 0.0

    max_dd = float(equity["drawdown"].max()) if "drawdown" in equity.columns else 0.0
    total_return = float(equity.iloc[-1]["equity"] / equity.iloc[0]["equity"] - 1.0) if len(equity) > 1 else 0.0
    calmar = (total_return / max_dd) if max_dd > 0 else 0.0

    if trades is None or trades.empty:
        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(max_dd),
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
        }

    gp = float(trades.loc[trades["net"] > 0, "net"].sum())
    gl = float(-trades.loc[trades["net"] <= 0, "net"].sum())
    pf = (gp / gl) if gl > 0 else float("inf")
    wr = float((trades["net"] > 0).mean()) if len(trades) else 0.0
    expectancy = float(trades["net"].mean()) if len(trades) else 0.0
    net = float(trades["net"].sum()) if len(trades) else 0.0
    recovery = (net / max_dd) if max_dd > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_dd),
        "profit_factor": float(pf),
        "win_rate": float(wr),
        "expectancy": float(expectancy),
        "recovery_factor": float(recovery),
    }


def prepare_full_dataset(root: Path, symbol: str, interval: str, cfg: dict) -> tuple[pd.DataFrame, str]:
    if symbol.upper() == GLOBAL_CONFIG["solusdt"]["symbol"]:
        return build_solusdt_dataset(
            root,
            SolusdtPipelineSpec(
                interval=interval,
                date_from=GLOBAL_CONFIG["solusdt"]["history_start"],
                date_to=GLOBAL_CONFIG["solusdt"]["history_end"],
            ),
        )

    spec = LoadSpec(symbol=symbol, interval=interval, date_from="2010-01-01", date_to="2100-01-01", market_type=cfg.get("market", {}).get("type", "spot"))
    df, source = load_ohlcv(root, spec)
    df = add_features(df)
    setups = primary_setups(df)
    df = pd.concat([df, setups], axis=1)
    h = int(GLOBAL_CONFIG["label_horizon_bars"])
    df["label_tb"] = triple_barrier_label(df, horizon=h, up_mult=1.5, dn_mult=1.0)
    df["label_fh"] = fixed_horizon_label(df, horizon=h, threshold=0.0)
    df["label"] = df["label_tb"].fillna(df["label_fh"])
    df["event_end_idx"] = np.minimum(np.arange(len(df)) + h, len(df) - 1)
    return df, source


def run_single_backtest(
    root: Path,
    symbol: str,
    interval: str,
    date_from: str,
    date_to: str,
    cash: float,
    prepared_df: Optional[pd.DataFrame] = None,
    prepared_source: Optional[str] = None,
):
    cfg = yaml.safe_load((root / "config.yaml").read_text())

    if prepared_df is None:
        df, source = prepare_full_dataset(root, symbol, interval, cfg)
    else:
        source = prepared_source or "prepared_cache"
        df = prepared_df.copy()

    d0 = pd.Timestamp(date_from, tz="UTC")
    d1 = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.ts >= d0) & (df.ts < d1)].copy()

    # avoid recompute if already prepared by pipeline/strategy
    if "regime" not in df.columns:
        df["regime"] = regime_rules(df)
    if not {"p_stay_bull", "p_stay_bear", "p_stay_range"}.issubset(set(df.columns)):
        trans = regime_transition_prob(df["regime"]).fillna(0.33)
        df = pd.concat([df, trans], axis=1)

    feats = feature_columns(df)
    ml = df.dropna(subset=feats + ["label", "setup_any", "event_end_idx"]).copy()
    ml = ml[ml["setup_any"] == 1].copy().reset_index(drop=True)

    # speed guardrail for ultra benchmark: keep most recent rows only
    max_rows = int(GLOBAL_CONFIG.get("max_ml_rows_per_strategy", 0) or 0)
    if max_rows > 0 and len(ml) > max_rows:
        ml = ml.iloc[-max_rows:].reset_index(drop=True)

    leakage_warnings = _warn_leakage(ml)

    fallback_trade_flag = df.get("trade_flag", df.get("setup_any", 0)).fillna(0).astype(int).copy()
    df["meta_proba"] = 0.0
    df["model_trade_flag"] = 0

    outer_logs: List[Dict] = []
    ts_to_df_idx = df.reset_index().set_index("ts")["index"]
    if len(ml) >= GLOBAL_CONFIG["min_train_rows"] + GLOBAL_CONFIG["min_test_rows"] and ml["label"].nunique() > 1:
        outer = _chrono_slices(len(ml), GLOBAL_CONFIG["outer_splits"])
        for fold, (tr_idx, te_idx) in enumerate(outer, start=1):
            tr_idx_safe = _purge_embargo_train_indices(
                tr_idx,
                te_idx,
                event_end_idx=ml["event_end_idx"].astype(int).values,
                embargo_bars=GLOBAL_CONFIG["embargo_bars"],
            )
            if len(tr_idx_safe) < GLOBAL_CONFIG["min_train_rows"] or len(te_idx) < GLOBAL_CONFIG["min_test_rows"]:
                continue

            Xtr = ml.iloc[tr_idx_safe][feats]
            ytr = ml.iloc[tr_idx_safe]["label"].astype(int)
            Xte = ml.iloc[te_idx][feats]
            yte = ml.iloc[te_idx]["label"].astype(int)
            if ytr.nunique() < 2 or yte.nunique() < 2:
                continue

            model, model_name, thr, inner_logs = _nested_select(
                X=Xtr,
                y=ytr,
                event_end_idx=ml.iloc[tr_idx_safe]["event_end_idx"].astype(int).values,
            )

            pte = model.predict_proba(Xte)[:, 1]
            oos_auc = float(roc_auc_score(yte, pte)) if yte.nunique() > 1 else 0.5

            # OOS only predictions written on test slice
            df_idx = ml.iloc[te_idx]["ts"].map(ts_to_df_idx).values
            for pos, di in enumerate(df_idx):
                if pd.notna(di):
                    dfi = int(di)
                    df.at[df.index[dfi], "meta_proba"] = float(pte[pos])
            outer_logs.append({
                "fold": fold,
                "model": model_name,
                "threshold": float(thr),
                "oos_auc": oos_auc,
                "train_rows": int(len(Xtr)),
                "test_rows": int(len(Xte)),
                "inner": inner_logs,
            })

            # threshold applied only on current OOS test block
            test_mask = df.index.isin(df.index[df_idx.astype(int)])
            df.loc[test_mask, "model_trade_flag"] = final_trade_flag(df.loc[test_mask], "meta_proba", thr)

    if outer_logs:
        df["trade_flag"] = df["model_trade_flag"].fillna(0).astype(int)
    else:
        # If no valid OOS folds, keep strategy-native signal instead of forcing flat portfolio.
        df["trade_flag"] = fallback_trade_flag

    tdf, edf = simulate_spot_long_only(df, cfg, initial_cash=float(cash))

    metrics = _oos_metrics(tdf, edf)
    summary = {
        "symbol": symbol,
        "interval": interval,
        "date_from": date_from,
        "date_to": date_to,
        "data_source": source,
        "reporting_scope": "OOS_TEST_ONLY",
        "anti_overfitting": {
            "purge_embargo": True,
            "embargo_bars": int(GLOBAL_CONFIG["embargo_bars"]),
            "nested_cv": True,
            "threshold_on_validation_only": True,
            "threshold_on_test": False,
        },
        "leakage_warnings": leakage_warnings,
        "outer_folds": outer_logs,
        "trades": int(len(tdf)) if isinstance(tdf, pd.DataFrame) else 0,
        **metrics,
    }

    return summary, tdf, edf, df


def save_backtest_outputs(root: Path, out_dir: str, summary: dict, trades: pd.DataFrame, equity: pd.DataFrame):
    out = (root / out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    trade_cols = ["entry_ts", "exit_ts", "entry", "exit", "qty", "gross", "fees", "net", "reason"]
    eq_cols = ["ts", "equity", "drawdown", "position_qty"]

    trades_safe = trades.reindex(columns=trade_cols) if isinstance(trades, pd.DataFrame) else pd.DataFrame(columns=trade_cols)
    equity_safe = equity.reindex(columns=eq_cols) if isinstance(equity, pd.DataFrame) else pd.DataFrame(columns=eq_cols)

    trades_safe.to_csv(out / "trades.csv", index=False)
    equity_safe.to_csv(out / "equity.csv", index=False)
