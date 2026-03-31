from __future__ import annotations

import pandas as pd


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def trix(close: pd.Series, length: int = 9, signal: int = 21) -> pd.DataFrame:
    e1 = _ema(close, length)
    e2 = _ema(e1, length)
    e3 = _ema(e2, length)
    trix_line = e3.pct_change() * 100.0
    trix_signal = _ema(trix_line, signal)
    trix_hist = trix_line - trix_signal
    return pd.DataFrame({
        "trix": trix_line,
        "trix_signal": trix_signal,
        "trix_hist": trix_hist,
    })


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ma_dn = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ma_up / ma_dn.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def stoch_rsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    r = rsi(close, rsi_len)
    mn = r.rolling(stoch_len).min()
    mx = r.rolling(stoch_len).max()
    stoch = (r - mn) / (mx - mn).replace(0, pd.NA)
    k_line = (stoch * 100).rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return pd.DataFrame({
        "stoch_rsi_k": k_line,
        "stoch_rsi_d": d_line,
    })


def add_trix_stochrsi_combo(
    df: pd.DataFrame,
    trix_len: int = 9,
    trix_signal: int = 21,
    rsi_len: int = 14,
    stoch_len: int = 14,
    k: int = 3,
    d: int = 3,
    stoch_oversold: float = 20.0,
    stoch_overbought: float = 80.0,
) -> pd.DataFrame:
    """
    Adds a combined indicator + basic long/flat signals.

    Long entry condition (combo_long):
    - TRIX line above TRIX signal (trend momentum up)
    - Stoch RSI K crosses above D
    - K < oversold threshold recently / or under 50 to avoid late entries

    Exit condition (combo_exit):
    - TRIX line below signal OR
    - Stoch RSI K crosses below D in high zone
    """
    out = df.copy()
    t = trix(out["close"], trix_len, trix_signal)
    s = stoch_rsi(out["close"], rsi_len, stoch_len, k, d)
    out = pd.concat([out, t, s], axis=1)

    k_now = out["stoch_rsi_k"]
    d_now = out["stoch_rsi_d"]
    k_prev = k_now.shift(1)
    d_prev = d_now.shift(1)

    cross_up = (k_prev <= d_prev) & (k_now > d_now)
    cross_dn = (k_prev >= d_prev) & (k_now < d_now)

    trix_bull = out["trix"] > out["trix_signal"]
    trix_bear = out["trix"] < out["trix_signal"]

    stoch_ok_entry = (k_now < 50) | (k_now.rolling(5).min() <= stoch_oversold)
    stoch_risk_exit = (k_now >= stoch_overbought) & cross_dn

    out["combo_long"] = (trix_bull & cross_up & stoch_ok_entry).astype(int)
    out["combo_exit"] = (trix_bear | stoch_risk_exit).astype(int)

    return out
