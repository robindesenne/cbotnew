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


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def supertrend(df: pd.DataFrame, period: int = 10, factor: float = 3.0) -> pd.DataFrame:
    d = df.copy()
    hl2 = (d["high"] + d["low"]) / 2.0
    atr_v = atr(d, period)
    upper = hl2 + factor * atr_v
    lower = hl2 - factor * atr_v

    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(d)):
        if (upper.iloc[i] < final_upper.iloc[i - 1]) or (d["close"].iloc[i - 1] > final_upper.iloc[i - 1]):
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (lower.iloc[i] > final_lower.iloc[i - 1]) or (d["close"].iloc[i - 1] < final_lower.iloc[i - 1]):
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    st = pd.Series(index=d.index, dtype=float)
    dir_up = pd.Series(index=d.index, dtype=bool)

    for i in range(1, len(d)):
        prev_st = st.iloc[i - 1] if i > 0 else np.nan
        if np.isnan(prev_st):
            if d["close"].iloc[i] <= final_upper.iloc[i]:
                st.iloc[i] = final_upper.iloc[i]
                dir_up.iloc[i] = False
            else:
                st.iloc[i] = final_lower.iloc[i]
                dir_up.iloc[i] = True
            continue

        if prev_st == final_upper.iloc[i - 1]:
            if d["close"].iloc[i] <= final_upper.iloc[i]:
                st.iloc[i] = final_upper.iloc[i]
                dir_up.iloc[i] = False
            else:
                st.iloc[i] = final_lower.iloc[i]
                dir_up.iloc[i] = True
        else:
            if d["close"].iloc[i] >= final_lower.iloc[i]:
                st.iloc[i] = final_lower.iloc[i]
                dir_up.iloc[i] = True
            else:
                st.iloc[i] = final_upper.iloc[i]
                dir_up.iloc[i] = False

    out = pd.DataFrame({
        "supertrend": st,
        "supertrend_up": dir_up.fillna(False),
        "supertrend_down": (~dir_up).fillna(False),
    }, index=d.index)
    return out


def add_indicators(df: pd.DataFrame, st_period: int, st_factor: float) -> pd.DataFrame:
    d = df.copy()
    st = supertrend(d, st_period, st_factor)
    d = pd.concat([d, st], axis=1)
    d["adx14"] = adx(d, 14)
    d["rsi14"] = rsi(d["close"], 14)
    d["vol_ma20"] = d["volume"].rolling(20).mean()

    # 1H EMA200 filter
    h = d.set_index("ts").resample("1H").agg({"close": "last"}).dropna().reset_index()
    h["ema200_1h"] = ema(h["close"], 200)
    d = pd.merge_asof(d.sort_values("ts"), h[["ts", "ema200_1h"]].sort_values("ts"), on="ts", direction="backward")

    d["st_flip_up"] = (~d["supertrend_up"].shift(1).fillna(False)) & (d["supertrend_up"])
    d["st_flip_dn"] = (~d["supertrend_down"].shift(1).fillna(False)) & (d["supertrend_down"])
    return d


def run_bt(df: pd.DataFrame, cash0: float, risk_per_trade: float, fee: float, slip_bps: float, adx_th: float):
    slip = slip_bps / 10000.0
    cash = cash0
    peak = cash
    pos = None
    trades = []
    eq = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        mark = float(row["close"])
        equity = cash if pos is None else cash + (mark - pos["entry"]) * pos["qty"] * pos["side"]
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak else 0
        eq.append({"ts": row["ts"], "equity": equity, "drawdown": dd})

        if pos is not None:
            # Trailing stop on supertrend line
            st_line = float(row["supertrend"]) if pd.notna(row["supertrend"]) else pos["stop"]
            if pos["side"] == 1:
                pos["stop"] = max(pos["stop"], st_line)
                stop_hit = row["low"] <= pos["stop"]
                tp_hit = row["high"] >= pos["tp"]
                flip_exit = bool(row["st_flip_dn"])
            else:
                pos["stop"] = min(pos["stop"], st_line)
                stop_hit = row["high"] >= pos["stop"]
                tp_hit = row["low"] <= pos["tp"]
                flip_exit = bool(row["st_flip_up"])

            if stop_hit or tp_hit or flip_exit:
                if stop_hit:
                    exit_px = pos["stop"]
                    reason = "stop"
                elif tp_hit:
                    exit_px = pos["tp"]
                    reason = "tp"
                else:
                    exit_px = float(nxt["open"])
                    reason = "flip"

                exit_px = exit_px * (1 - slip * pos["side"])
                gross = (exit_px - pos["entry"]) * pos["qty"] * pos["side"]
                fees = (abs(pos["entry"] * pos["qty"]) + abs(exit_px * pos["qty"])) * fee
                net = gross - fees
                cash += net
                trades.append({
                    "entry_ts": pos["entry_ts"], "exit_ts": row["ts"], "side": "LONG" if pos["side"] == 1 else "SHORT",
                    "entry": pos["entry"], "exit": exit_px, "qty": pos["qty"], "gross": gross, "fees": fees, "net": net, "reason": reason,
                })
                pos = None

        if pos is None:
            vol_ok = row["volume"] > row["vol_ma20"] if pd.notna(row["vol_ma20"]) else False

            long_sig = (
                bool(row["supertrend_up"]) and (row["close"] > row["supertrend"]) and
                (row["close"] > row["ema200_1h"]) and (row["adx14"] > adx_th) and
                (row["rsi14"] > 45) and vol_ok
            )
            short_sig = (
                bool(row["supertrend_down"]) and (row["close"] < row["supertrend"]) and
                (row["close"] < row["ema200_1h"]) and (row["adx14"] > adx_th) and
                (row["rsi14"] < 55) and vol_ok
            )

            if long_sig or short_sig:
                side = 1 if long_sig else -1
                entry = float(nxt["open"]) * (1 + slip * side)
                st_line = float(row["supertrend"]) if pd.notna(row["supertrend"]) else np.nan
                if not np.isfinite(st_line):
                    continue
                stop_dist = abs(entry - st_line)
                if stop_dist <= 0:
                    continue

                risk_cash = cash * risk_per_trade
                qty = risk_cash / stop_dist
                if qty <= 0:
                    continue

                if side == 1:
                    stop = st_line
                    tp = entry + (2.0 * stop_dist)
                else:
                    stop = st_line
                    tp = entry - (2.0 * stop_dist)

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
    ap.add_argument("--risk", type=float, default=0.0075)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--out", default="reports/v2pro_supertrend")
    args = ap.parse_args()

    df, source = load_ohlcv(ROOT, LoadSpec(symbol=args.symbol, interval=args.interval, date_from=args.date_from, date_to=args.date_to, market_type="spot"))

    results = []
    best = None
    best_pack = None

    for p in [9, 10, 11, 12]:
        for f in [2.5, 3.0, 3.5]:
            d = add_indicators(df, p, f)
            for adx_th in [20, 21, 22, 23, 24, 25]:
                met, tdf, eqdf = run_bt(d, args.cash, args.risk, args.fee, args.slip_bps, adx_th)
                row = {"st_period": p, "st_factor": f, "adx_th": adx_th, **met}
                results.append(row)
                score = (met["return_pct"], met["profit_factor"], -met["max_drawdown"])
                if best is None or score > best:
                    best = score
                    best_pack = (p, f, adx_th, met, tdf, eqdf)

    p, f, adx_th, metrics, tdf, eqdf = best_pack
    monthly = monthly_perf(eqdf)

    out = ROOT / args.out / f"{args.symbol}_{args.interval}_{args.date_from}_{args.date_to}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values(["return_pct", "profit_factor"], ascending=False).to_csv(out / "grid_results.csv", index=False)
    tdf.to_csv(out / "trades.csv", index=False)
    eqdf.to_csv(out / "equity.csv", index=False)
    monthly.to_csv(out / "monthly_performance.csv", index=False)

    prev_summary = None
    prev_path = ROOT / "reports" / "v2pro_macd_ema_rsi_vwap" / f"{args.symbol}_{args.interval}_{args.date_from}_{args.date_to}" / "summary.json"
    if prev_path.exists():
        prev_summary = json.loads(prev_path.read_text())

    summary = {
        "symbol": args.symbol,
        "interval": args.interval,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "data_source": source,
        "strategy": "Supertrend + EMA200(1H) + ADX + RSI + VolumeMA20",
        "best_params": {"supertrend_period": p, "supertrend_factor": f, "adx_threshold": adx_th},
        "risk_per_trade": args.risk,
        "fee": args.fee,
        "slippage_bps": args.slip_bps,
        **metrics,
        "compare_prev_strategy": {
            "name": "EMA9/21 + MACD + RSI + VWAP",
            "summary": prev_summary,
        },
        "out_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
