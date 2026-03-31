from __future__ import annotations

import pandas as pd


def simulate_spot_long_only(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ex = cfg["exchange"]
    rk = cfg["risk"]
    st = cfg["strategy"]

    fee = float(ex["taker_fee"])
    slip = float(ex["slippage_bps"]) / 10000.0

    cash = 10000.0
    pos = None
    peak = cash
    daily_start = cash
    day = None
    cooldown = 0

    trades = []
    eq = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        if day != row.ts.date():
            day = row.ts.date()
            daily_start = cash if pos is None else cash + pos["qty"] * row.close

        equity = cash if pos is None else cash + pos["qty"] * row.close
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak else 0
        eq.append({"ts": row.ts, "equity": equity, "drawdown": dd})

        if cooldown > 0:
            cooldown -= 1

        if equity < daily_start * (1 - rk["daily_loss_limit_pct"]):
            continue

        if pos is not None:
            # conservative: if stop and target same bar => stop first
            stop_hit = row.low <= pos["stop"]
            tp_hit = row.high >= pos["tp"]
            exit_px = None
            reason = None
            if stop_hit:
                exit_px = pos["stop"] * (1 - slip)
                reason = "stop"
            elif tp_hit:
                exit_px = pos["tp"] * (1 - slip)
                reason = "tp"
            elif pos["bars"] >= st["max_hold_bars"]:
                exit_px = nxt.open * (1 - slip)
                reason = "time"

            if exit_px is not None:
                gross = (exit_px - pos["entry"]) * pos["qty"]
                fees = (pos["entry"] * pos["qty"] + exit_px * pos["qty"]) * fee
                net = gross - fees
                cash += exit_px * pos["qty"] - exit_px * pos["qty"] * fee
                trades.append({
                    "entry_ts": pos["entry_ts"], "exit_ts": row.ts,
                    "entry": pos["entry"], "exit": exit_px,
                    "qty": pos["qty"], "gross": gross, "fees": fees, "net": net,
                    "reason": reason,
                })
                if net <= 0:
                    cooldown = max(cooldown, int(rk["cooldown_bars_after_loss"]))
                pos = None
            else:
                pos["bars"] += 1

        if pos is None and cooldown == 0 and int(row["trade_flag"]) == 1:
            entry = nxt.open * (1 + slip)
            atr = max(row.atr_14, entry * rk["min_stop_distance_bps"] / 10000)
            stop = entry - st["stop_atr_mult"] * atr
            risk_per_unit = max(entry - stop, 1e-9)

            risk_cash = equity * rk["risk_per_trade"]
            qty_risk = risk_cash / risk_per_unit
            qty_cash = (equity * rk["max_position_pct"]) / entry
            qty = min(qty_risk, qty_cash)

            if qty * entry >= rk["min_notional"]:
                cost = qty * entry * (1 + fee)
                if cost <= cash:
                    cash -= cost
                    tp = entry + st["tp_r_multiple"] * risk_per_unit
                    pos = {"entry_ts": nxt.ts, "entry": entry, "qty": qty, "stop": stop, "tp": tp, "bars": 0}

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(eq)
    return tdf, edf
