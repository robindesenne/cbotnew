from __future__ import annotations

import pandas as pd


def _annual_factor(interval: str) -> float:
    m = int(interval[:-1]) * {"m": 1, "h": 60, "d": 1440}[interval[-1]]
    return ((365 * 24 * 60) / m) ** 0.5


def simulate_v2(seg: pd.DataFrame, cfg: dict, interval: str, initial_cash: float = 10000.0):
    ex = cfg.get("exchange", {})
    rk = cfg.get("risk", {})
    st = cfg.get("strategy", {})

    fee = float(ex.get("taker_fee", 0.001))
    slip = float(ex.get("slippage_bps", 2.0)) / 10000.0

    cash = initial_cash
    pos = None
    peak = cash
    eq_rows, trades = [], []

    for i in range(1, len(seg) - 1):
        row = seg.iloc[i]
        nxt = seg.iloc[i + 1]

        eq = cash if pos is None else cash + pos["qty"] * row.close
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak else 0
        eq_rows.append({"ts": row.ts, "equity": eq, "drawdown": dd, "position_qty": 0 if pos is None else pos["qty"]})

        if pos is not None:
            stop_hit = row.low <= pos["stop"]
            tp_hit = row.high >= pos["tp"]
            timeout = pos["bars"] >= int(st.get("max_hold_bars", 64))
            if stop_hit or tp_hit or timeout:
                exit_px = pos["stop"] if stop_hit else (pos["tp"] if tp_hit else nxt.open)
                exit_px *= (1 - slip)
                gross = (exit_px - pos["entry"]) * pos["qty"]
                fees = (pos["entry"] * pos["qty"] + exit_px * pos["qty"]) * fee
                net = gross - fees
                cash += exit_px * pos["qty"] - exit_px * pos["qty"] * fee
                trades.append({"entry_ts": pos["entry_ts"], "exit_ts": row.ts, "entry": pos["entry"], "exit": exit_px, "qty": pos["qty"], "gross": gross, "fees": fees, "net": net, "reason": "stop" if stop_hit else ("tp" if tp_hit else "time")})
                pos = None
            else:
                pos["bars"] += 1

        if pos is None and int(row.get("trade_flag", 0)) == 1:
            entry = float(nxt.open) * (1 + slip)
            atr = max(float(row.atr_14), entry * float(rk.get("min_stop_distance_bps", 15)) / 10000.0)
            stop = entry - float(st.get("stop_atr_mult", 1.5)) * atr
            risk_per_unit = max(entry - stop, 1e-9)

            # V2 dynamic sizing: risk cap + vol targeting proxy
            ann_vol = max(float(row.get("rv_24", 0.01) if pd.notna(row.get("rv_24", 0.01)) else 0.01) * _annual_factor(interval), 1e-4)
            vol_target = float(rk.get("vol_target_ann", 0.18))
            vol_scale = min(2.0, max(0.4, vol_target / ann_vol))

            risk_cash = eq * min(float(rk.get("risk_per_trade", 0.005)) * vol_scale, float(rk.get("risk_per_trade_max", 0.03)))
            qty_risk = risk_cash / risk_per_unit
            qty_cap = (eq * float(rk.get("max_position_pct", 0.25))) / entry
            qty = min(qty_risk, qty_cap)

            if qty * entry >= float(rk.get("min_notional", 10)):
                cost = qty * entry * (1 + fee)
                if cost <= cash:
                    cash -= cost
                    tp = entry + float(st.get("tp_r_multiple", 2.2)) * risk_per_unit
                    pos = {"entry_ts": nxt.ts, "entry": entry, "qty": qty, "stop": stop, "tp": tp, "bars": 0}

    return pd.DataFrame(trades), pd.DataFrame(eq_rows)
