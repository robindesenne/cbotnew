#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))
from src_v2.data_loader import LoadSpec, load_ohlcv


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ma_up / ma_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema9"] = ema(d["close"], 9)
    d["ema21"] = ema(d["close"], 21)

    macd_fast = ema(d["close"], 12)
    macd_slow = ema(d["close"], 26)
    d["macd"] = macd_fast - macd_slow
    d["macd_signal"] = ema(d["macd"], 9)

    d["rsi14"] = rsi(d["close"], 14)
    d["atr14"] = atr(d, 14)

    day = d["ts"].dt.floor("D")
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    d["vwap"] = (tp * d["volume"]).groupby(day).cumsum() / d["volume"].groupby(day).cumsum().replace(0, np.nan)

    d["macd_xup"] = (d["macd"].shift(1) <= d["macd_signal"].shift(1)) & (d["macd"] > d["macd_signal"])
    d["macd_xdn"] = (d["macd"].shift(1) >= d["macd_signal"].shift(1)) & (d["macd"] < d["macd_signal"])
    return d


def run_bt(df: pd.DataFrame, cash0: float, risk_per_trade: float, fee: float, slip_bps: float, sl_pct: float, tp_pct: float, use_atr_stop: bool):
    slip = slip_bps / 10000.0
    cash = cash0
    peak = cash
    pos = None
    trades = []
    eq = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        mark = row["close"]
        equity = cash if pos is None else cash + (pos["qty"] * (mark - pos["entry"]) * pos["side"])
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak else 0
        eq.append({"ts": row["ts"], "equity": equity, "drawdown": dd})

        if pos is not None:
            stop_hit = False
            tp_hit = False
            if pos["side"] == 1:
                stop_hit = row["low"] <= pos["stop"]
                tp_hit = row["high"] >= pos["tp"]
                exit_px = pos["stop"] if stop_hit else (pos["tp"] if tp_hit else None)
            else:
                stop_hit = row["high"] >= pos["stop"]
                tp_hit = row["low"] <= pos["tp"]
                exit_px = pos["stop"] if stop_hit else (pos["tp"] if tp_hit else None)

            if exit_px is not None:
                exit_px = exit_px * (1 - slip * pos["side"])
                gross = (exit_px - pos["entry"]) * pos["qty"] * pos["side"]
                fees = (abs(pos["entry"] * pos["qty"]) + abs(exit_px * pos["qty"])) * fee
                net = gross - fees
                cash += net
                trades.append({"entry_ts": pos["entry_ts"], "exit_ts": row["ts"], "side": "LONG" if pos["side"] == 1 else "SHORT", "entry": pos["entry"], "exit": exit_px, "qty": pos["qty"], "gross": gross, "fees": fees, "net": net, "reason": "stop" if stop_hit else "tp"})
                pos = None

        if pos is None:
            long_sig = (row["ema9"] > row["ema21"]) and bool(row["macd_xup"]) and (row["rsi14"] < 70) and (row["close"] > row["vwap"])
            short_sig = (row["ema9"] < row["ema21"]) and bool(row["macd_xdn"]) and (row["rsi14"] > 30) and (row["close"] < row["vwap"])

            if long_sig or short_sig:
                side = 1 if long_sig else -1
                entry = float(nxt["open"]) * (1 + slip * side)
                stop_dist = max(entry * sl_pct, float(row["atr14"]) if use_atr_stop and pd.notna(row["atr14"]) else 0.0)
                if stop_dist <= 0:
                    continue
                risk_cash = cash * risk_per_trade
                qty = risk_cash / stop_dist
                if qty <= 0:
                    continue

                if side == 1:
                    stop = entry - stop_dist
                    tp = entry + max(entry * tp_pct, stop_dist * 1.7)
                else:
                    stop = entry + stop_dist
                    tp = entry - max(entry * tp_pct, stop_dist * 1.7)

                pos = {"side": side, "entry": entry, "qty": qty, "stop": stop, "tp": tp, "entry_ts": nxt["ts"]}

    tdf = pd.DataFrame(trades)
    eqdf = pd.DataFrame(eq)
    net = float(tdf["net"].sum()) if len(tdf) else 0.0
    wins = int((tdf["net"] > 0).sum()) if len(tdf) else 0
    gp = float(tdf.loc[tdf.net > 0, "net"].sum()) if len(tdf) else 0.0
    gl = float(-tdf.loc[tdf.net <= 0, "net"].sum()) if len(tdf) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    mdd = float(eqdf["drawdown"].max()) if len(eqdf) else 0.0

    return {
        "trades": int(len(tdf)),
        "win_rate": float(wins / len(tdf)) if len(tdf) else 0.0,
        "profit_factor": float(pf),
        "max_drawdown": mdd,
        "net_pnl": net,
        "return_pct": float(net / cash0),
    }, tdf, eqdf


def monthly_perf(eqdf: pd.DataFrame) -> pd.DataFrame:
    if eqdf.empty:
        return pd.DataFrame(columns=["month", "equity_start", "equity_end", "monthly_return_pct"])
    d = eqdf.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["month"] = d["ts"].dt.to_period("M").astype(str)
    g = d.groupby("month", as_index=False).agg(equity_start=("equity", "first"), equity_end=("equity", "last"))
    g["monthly_return_pct"] = (g["equity_end"] / g["equity_start"] - 1.0) * 100.0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--cash", type=float, default=10000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--fee", type=float, default=0.0004, help="0.0004 spot or 0.0002 futures with BNB")
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--out", default="reports/v2pro_macd_ema_rsi_vwap")
    args = ap.parse_args()

    df, source = load_ohlcv(ROOT, LoadSpec(symbol=args.symbol, interval=args.interval, date_from=args.date_from, date_to=args.date_to, market_type="spot"))
    df = add_indicators(df)

    grid = [(0.003, 0.005)]
    if args.optimize:
        grid = [(s, t) for s in [0.0025, 0.0030, 0.0035] for t in [0.0045, 0.0050, 0.0060]]

    best = None
    best_pack = None
    for slp, tpp in grid:
        metrics, tdf, eqdf = run_bt(df, args.cash, args.risk, args.fee, args.slip_bps, slp, tpp, use_atr_stop=True)
        score = (metrics["return_pct"], metrics["profit_factor"], -metrics["max_drawdown"])
        if best is None or score > best:
            best = score
            best_pack = (slp, tpp, metrics, tdf, eqdf)

    slp, tpp, metrics, tdf, eqdf = best_pack
    monthly = monthly_perf(eqdf)

    out = ROOT / args.out / f"{args.symbol}_{args.interval}_{args.date_from}_{args.date_to}"
    out.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(out / "trades.csv", index=False)
    eqdf.to_csv(out / "equity.csv", index=False)
    monthly.to_csv(out / "monthly_performance.csv", index=False)

    summary = {
        "symbol": args.symbol,
        "interval": args.interval,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "data_source": source,
        "strategy": "EMA9/EMA21 + MACD(12,26,9) crossover + RSI14 + VWAP",
        "sl_pct": slp,
        "tp_pct": tpp,
        "use_atr_stop": True,
        "risk_per_trade": args.risk,
        "fee": args.fee,
        "slippage_bps": args.slip_bps,
        **metrics,
        "out_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
