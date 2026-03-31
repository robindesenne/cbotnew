from __future__ import annotations

import pandas as pd


def walkforward_windows(date_from: str, date_to: str, train_days: int, cal_days: int, test_days: int, step_days: int):
    d0 = pd.Timestamp(date_from, tz="UTC")
    d1 = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
    cur = d0
    while cur + pd.Timedelta(days=train_days + cal_days + test_days) <= d1:
        tr_start = cur
        tr_end = tr_start + pd.Timedelta(days=train_days)
        cal_end = tr_end + pd.Timedelta(days=cal_days)
        te_end = cal_end + pd.Timedelta(days=test_days)
        yield tr_start, tr_end, cal_end, te_end
        cur += pd.Timedelta(days=step_days)


def apply_purge_embargo(ml: pd.DataFrame, tr_mask, cal_mask, te_start, te_end, horizon_bars: int, interval_minutes: int):
    embargo = pd.Timedelta(minutes=interval_minutes * horizon_bars)
    event_end_ts = ml["ts"].shift(-horizon_bars).ffill()
    te_start_emb = te_start - embargo
    te_end_emb = te_end + embargo
    overlap = (event_end_ts >= te_start_emb) & (ml["ts"] <= te_end_emb)
    return tr_mask & (~overlap), cal_mask & (~overlap)
