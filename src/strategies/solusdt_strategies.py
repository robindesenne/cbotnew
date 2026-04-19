from __future__ import annotations

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, StrategyContext
from .strategy_registry import StrategySpec, registry


def _finalize(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    out = df.copy()
    s = signal.fillna(0).clip(-1, 1).astype(int)
    out["signal"] = s

    # Global long-only quality gates to reduce noisy entries
    trend_ok = pd.Series(True, index=out.index)
    if "ema_200" in out.columns:
        trend_ok = out["close"] > out["ema_200"]

    vol_ok = pd.Series(True, index=out.index)
    if "atr_pct" in out.columns:
        vol_cap = out["atr_pct"].rolling(200, min_periods=50).quantile(0.90)
        vol_ok = out["atr_pct"] <= vol_cap

    liq_ok = pd.Series(True, index=out.index)
    if "rel_volume" in out.columns:
        liq_ok = out["rel_volume"].fillna(0) >= 0.65

    long_gate = trend_ok.fillna(False) & vol_ok.fillna(False) & liq_ok.fillna(False)
    out["trade_flag"] = ((s == 1) & long_gate).astype(int)  # long-only executor compatibility
    out["meta_proba"] = np.where(s == 1, 0.68, np.where(s == -1, 0.32, 0.50))
    return out


# =========================
# 1) Momentum (5)
# =========================
class MomentumBaseStrategy(BaseStrategy):
    family = "momentum"

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        x = self.prepare_dataset(df, ctx)
        fast = int(self.params.get("fast", 20))
        slow = int(self.params.get("slow", 50))
        mode = self.params.get("mode", "ema_cross")

        if mode == "ema_cross":
            ema_fast = x[f"ema_{fast}"] if f"ema_{fast}" in x.columns else x["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = x[f"ema_{slow}"] if f"ema_{slow}" in x.columns else x["close"].ewm(span=slow, adjust=False).mean()
            sig = np.where(ema_fast > ema_slow, 1, -1)
        elif mode == "ema_cross_event":
            ema_fast = x[f"ema_{fast}"] if f"ema_{fast}" in x.columns else x["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = x[f"ema_{slow}"] if f"ema_{slow}" in x.columns else x["close"].ewm(span=slow, adjust=False).mean()
            cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
            sig = np.where(cross_up, 1, 0)
        elif mode == "ema_cross_event_vol":
            ema_fast = x[f"ema_{fast}"] if f"ema_{fast}" in x.columns else x["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = x[f"ema_{slow}"] if f"ema_{slow}" in x.columns else x["close"].ewm(span=slow, adjust=False).mean()
            cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
            vol_ok = (x["rv_24"] < x["rv_96"]) if {"rv_24", "rv_96"}.issubset(x.columns) else pd.Series(True, index=x.index)
            sig = np.where(cross_up & vol_ok.fillna(False), 1, 0)
        elif mode == "ema_cross_event_vol_dist":
            ema_fast = x[f"ema_{fast}"] if f"ema_{fast}" in x.columns else x["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = x[f"ema_{slow}"] if f"ema_{slow}" in x.columns else x["close"].ewm(span=slow, adjust=False).mean()
            cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
            vol_ok = (x["rv_24"] < x["rv_96"]) if {"rv_24", "rv_96"}.issubset(x.columns) else pd.Series(True, index=x.index)
            dist_min = float(self.params.get("dist_min", 0.01))
            trend_strength_ok = (x["dist_ema_200"] >= dist_min) if "dist_ema_200" in x.columns else pd.Series(True, index=x.index)
            sig = np.where(cross_up & vol_ok.fillna(False) & trend_strength_ok.fillna(False), 1, 0)
        elif mode == "ema_cross_event_vol_slope":
            ema_fast = x[f"ema_{fast}"] if f"ema_{fast}" in x.columns else x["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = x[f"ema_{slow}"] if f"ema_{slow}" in x.columns else x["close"].ewm(span=slow, adjust=False).mean()
            cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
            vol_ok = (x["rv_24"] < x["rv_96"]) if {"rv_24", "rv_96"}.issubset(x.columns) else pd.Series(True, index=x.index)
            slope_min = float(self.params.get("slope_min", 0.0))
            trend_slope_ok = (x["slope_ema_50"] > slope_min) if "slope_ema_50" in x.columns else pd.Series(True, index=x.index)
            sig = np.where(cross_up & vol_ok.fillna(False) & trend_slope_ok.fillna(False), 1, 0)
        elif mode == "macd_like":
            ema_fast = x[f"ema_{fast}"] if f"ema_{fast}" in x.columns else x["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = x[f"ema_{slow}"] if f"ema_{slow}" in x.columns else x["close"].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            sigl = macd.ewm(span=9, adjust=False).mean()
            sig = np.where(macd > sigl, 1, -1)
        elif mode == "trix_like":
            e1 = x["close"].ewm(span=fast, adjust=False).mean()
            e2 = e1.ewm(span=fast, adjust=False).mean()
            e3 = e2.ewm(span=fast, adjust=False).mean()
            trix = e3.pct_change(1)
            sig = np.where(trix > 0, 1, -1)
        elif mode == "roc":
            roc = x["close"].pct_change(int(self.params.get("roc_h", 12)))
            sig = np.where(roc > 0, 1, -1)
        else:  # dual_mom
            sig = np.where((x["mom_12"] > 0) & (x["mom_48"] > 0) & (x["close"] > x["ema_100"]), 1, -1)

        return _finalize(x, pd.Series(sig, index=x.index))


class MomentumEMACrossV1(MomentumBaseStrategy): name, version, default_params = "momentum_ema_cross", "1.1", {"mode": "ema_cross", "fast": 50, "slow": 200}
class MomentumEMACrossEventV1(MomentumBaseStrategy): name, version, default_params = "momentum_ema_cross_event", "1.0", {"mode": "ema_cross_event", "fast": 50, "slow": 200}
class MomentumEMACrossEventVolV1(MomentumBaseStrategy): name, version, default_params = "momentum_ema_cross_event_vol", "1.0", {"mode": "ema_cross_event_vol", "fast": 50, "slow": 200}
class MomentumEMACrossEventVolDistV1(MomentumBaseStrategy): name, version, default_params = "momentum_ema_cross_event_vol_dist", "1.0", {"mode": "ema_cross_event_vol_dist", "fast": 50, "slow": 200, "dist_min": 0.01}
class MomentumEMACrossEventVolSlopeV1(MomentumBaseStrategy): name, version, default_params = "momentum_ema_cross_event_vol_slope", "1.0", {"mode": "ema_cross_event_vol_slope", "fast": 50, "slow": 200, "slope_min": 0.0}
class MomentumMACDV1(MomentumBaseStrategy): name, version, default_params = "momentum_macd", "1.1", {"mode": "macd_like", "fast": 12, "slow": 26}
class MomentumTrixV1(MomentumBaseStrategy): name, version, default_params = "momentum_trix", "1.1", {"mode": "trix_like", "fast": 24}
class MomentumROCV1(MomentumBaseStrategy): name, version, default_params = "momentum_roc", "1.1", {"mode": "roc", "roc_h": 24, "fast": 50, "slow": 200}
class MomentumDualV1(MomentumBaseStrategy): name, version, default_params = "momentum_dual", "1.1", {"mode": "dual"}


# =========================
# 2) Mean Reversion (5)
# =========================
class MeanRevBaseStrategy(BaseStrategy):
    family = "mean_reversion"

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        x = self.prepare_dataset(df, ctx)
        mode = self.params.get("mode", "rsi")

        if mode == "rsi":
            lo, hi = self.params.get("lo", 25), self.params.get("hi", 75)
            sig = np.where((x["rsi_14"] < lo) & (x["close"] > x["ema_200"]), 1,
                           np.where((x["rsi_14"] > hi) & (x["close"] < x["ema_200"]), -1, 0))
        elif mode == "bollinger":
            n = int(self.params.get("n", 20)); k = float(self.params.get("k", 2.0))
            ma = x["close"].rolling(n).mean(); sd = x["close"].rolling(n).std()
            low, up = ma - k * sd, ma + k * sd
            sig = np.where(x["close"] < low, 1, np.where(x["close"] > up, -1, 0))
        elif mode == "stoch":
            n = int(self.params.get("n", 14))
            ll = x["low"].rolling(n).min(); hh = x["high"].rolling(n).max()
            k = 100 * (x["close"] - ll) / (hh - ll).replace(0, np.nan)
            sig = np.where(k < 20, 1, np.where(k > 80, -1, 0))
        elif mode == "zscore":
            z = (x["close"] - x["close"].rolling(72).mean()) / x["close"].rolling(72).std().replace(0, np.nan)
            sig = np.where((z < -2.2) & (x["close"] > x["ema_200"]), 1,
                           np.where((z > 2.2) & (x["close"] < x["ema_200"]), -1, 0))
        else:  # drawdown bounce
            entry_dd = float(self.params.get("entry_dd", -0.09))
            exit_dd = float(self.params.get("exit_dd", -0.01))
            use_vol_filter = bool(self.params.get("use_vol_filter", False))
            vol_ok = (x["rv_24"] < x["rv_96"]) if ({"rv_24", "rv_96"}.issubset(x.columns) and use_vol_filter) else pd.Series(True, index=x.index)
            sig = np.where((x["local_drawdown_48"] < entry_dd) & (x["close"] > x["ema_200"]) & vol_ok.fillna(False), 1,
                           np.where(x["local_drawdown_48"] > exit_dd, -1, 0))

        return _finalize(x, pd.Series(sig, index=x.index))


class MeanRevRSIV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_rsi", "1.1", {"mode": "rsi", "lo": 25, "hi": 75}
class MeanRevBollingerV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_bollinger", "1.1", {"mode": "bollinger", "n": 30, "k": 2.2}
class MeanRevStochV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_stoch", "1.1", {"mode": "stoch", "n": 21}
class MeanRevZScoreV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_zscore", "1.1", {"mode": "zscore"}
class MeanRevDrawdownV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_drawdown_bounce", "1.1", {"mode": "drawdown"}
class MeanRevDrawdownLooseV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_drawdown_bounce_loose", "1.0", {"mode": "drawdown", "entry_dd": -0.07, "exit_dd": -0.01}
class MeanRevDrawdownDeepV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_drawdown_bounce_deep", "1.0", {"mode": "drawdown", "entry_dd": -0.12, "exit_dd": -0.01}
class MeanRevDrawdownLoose05V1(MeanRevBaseStrategy): name, version, default_params = "meanrev_drawdown_bounce_loose05", "1.0", {"mode": "drawdown", "entry_dd": -0.05, "exit_dd": -0.01}
class MeanRevDrawdownLooseVolV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_drawdown_bounce_loose_vol", "1.0", {"mode": "drawdown", "entry_dd": -0.07, "exit_dd": -0.01, "use_vol_filter": True}
class MeanRevDrawdownLoose05FastExitV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_drawdown_bounce_loose05_fastexit", "1.0", {"mode": "drawdown", "entry_dd": -0.05, "exit_dd": -0.03}
class MeanRevDrawdownLoose05SlowExitV1(MeanRevBaseStrategy): name, version, default_params = "meanrev_drawdown_bounce_loose05_slowexit", "1.0", {"mode": "drawdown", "entry_dd": -0.05, "exit_dd": -0.005}


# =========================
# 3) Breakout / Structure (5)
# =========================
class BreakoutBaseStrategy(BaseStrategy):
    family = "breakout_structure"

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        x = self.prepare_dataset(df, ctx)
        mode = self.params.get("mode", "donchian")

        if mode == "donchian":
            n = int(self.params.get("n", 55))
            hh = x["high"].rolling(n).max().shift(1)
            ll = x["low"].rolling(n).min().shift(1)
            sig = np.where((x["close"] > hh) & (x["close"] > x["ema_200"]), 1,
                           np.where((x["close"] < ll) & (x["close"] < x["ema_200"]), -1, 0))
        elif mode == "pivot":
            pp = (x["high"].shift(1) + x["low"].shift(1) + x["close"].shift(1)) / 3
            sig = np.where(x["close"] > pp * 1.002, 1, np.where(x["close"] < pp * 0.998, -1, 0))
        elif mode == "range_break":
            width = (x["hh_20"] - x["ll_20"]) / x["close"].replace(0, np.nan)
            sig = np.where((x["breakout_up"] == 1) & (width < 0.04) & (x["close"] > x["ema_100"]), 1,
                           np.where((x["breakout_dn"] == 1) & (x["close"] < x["ema_100"]), -1, 0))
        elif mode == "swing":
            swing_hi = x["high"].rolling(10).max().shift(1)
            swing_lo = x["low"].rolling(10).min().shift(1)
            sig = np.where(x["close"] > swing_hi, 1, np.where(x["close"] < swing_lo, -1, 0))
        else:  # orderblock-like proxy
            body = (x["close"] - x["open"]).abs()
            big = body > body.rolling(30).mean() * 1.5
            sig = np.where((x["close"] > x["ema_20"]) & big, 1, np.where((x["close"] < x["ema_20"]) & big, -1, 0))

        return _finalize(x, pd.Series(sig, index=x.index))


class BreakoutDonchianV1(BreakoutBaseStrategy): name, version, default_params = "breakout_donchian", "1.1", {"mode": "donchian", "n": 55}
class BreakoutPivotV1(BreakoutBaseStrategy): name, version, default_params = "breakout_pivot", "1.1", {"mode": "pivot"}
class BreakoutRangeV1(BreakoutBaseStrategy): name, version, default_params = "breakout_range", "1.1", {"mode": "range_break"}
class BreakoutSwingV1(BreakoutBaseStrategy): name, version, default_params = "breakout_swing", "1.1", {"mode": "swing"}
class BreakoutOrderBlockProxyV1(BreakoutBaseStrategy): name, version, default_params = "breakout_orderblock_proxy", "1.1", {"mode": "orderblock"}


# =========================
# 4) Volatility / Regime (5)
# =========================
class VolRegimeBaseStrategy(BaseStrategy):
    family = "volatility_regime"

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        x = self.prepare_dataset(df, ctx)
        mode = self.params.get("mode", "atr_expand")

        if mode == "atr_expand":
            sig = np.where((x["atr_pct"] > x["atr_pct"].rolling(50).mean()) & (x["mom_6"] > 0), 1, 0)
        elif mode == "keltner":
            mid = x["ema_20"]
            rng = 1.5 * x["atr_14"]
            sig = np.where(x["close"] > mid + rng, 1, np.where(x["close"] < mid - rng, -1, 0))
        elif mode == "rv_regime":
            sig = np.where((x["rv_24"] < x["rv_96"]) & (x["mom_12"] > 0), 1, np.where(x["rv_24"] > x["rv_96"] * 1.4, -1, 0))
        elif mode == "regime_trend_only":
            sig = np.where((x["regime"] == "bull") & (x["mom_6"] > 0), 1, np.where(x["regime"] == "bear", -1, 0))
        else:  # atr compression breakout
            comp = x["atr_pct"] < x["atr_pct"].rolling(80).quantile(0.25)
            sig = np.where(comp & (x["breakout_up"] == 1), 1, np.where(comp & (x["breakout_dn"] == 1), -1, 0))

        return _finalize(x, pd.Series(sig, index=x.index))


class VolAtrExpandV1(VolRegimeBaseStrategy): name, version, default_params = "vol_atr_expand", "1.1", {"mode": "atr_expand"}
class VolKeltnerV1(VolRegimeBaseStrategy): name, version, default_params = "vol_keltner", "1.1", {"mode": "keltner"}
class VolRvRegimeV1(VolRegimeBaseStrategy): name, version, default_params = "vol_rv_regime", "1.1", {"mode": "rv_regime"}
class VolTrendRegimeV1(VolRegimeBaseStrategy): name, version, default_params = "vol_regime_trend", "1.1", {"mode": "regime_trend_only"}
class VolCompressionBreakV1(VolRegimeBaseStrategy): name, version, default_params = "vol_compression_breakout", "1.1", {"mode": "compression"}


# =========================
# 5) Volume Profile / VWAP (5)
# =========================
class VolumeVwapBaseStrategy(BaseStrategy):
    family = "volume_vwap"

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        x = self.prepare_dataset(df, ctx)
        mode = self.params.get("mode", "vwap_cross")

        typical = (x["high"] + x["low"] + x["close"]) / 3
        vwap = (typical * x["volume"]).rolling(96).sum() / x["volume"].rolling(96).sum().replace(0, np.nan)

        if mode == "vwap_cross":
            sig = np.where((x["close"] > vwap) & (x["mom_6"] > 0), 1,
                           np.where((x["close"] < vwap) & (x["mom_6"] < 0), -1, 0))
        elif mode == "vwap_cross_event":
            cross_up = (x["close"] > vwap) & (x["close"].shift(1) <= vwap.shift(1))
            sig = np.where(cross_up & (x["mom_6"] > 0), 1, 0)
        elif mode == "vwap_cross_event_vol":
            cross_up = (x["close"] > vwap) & (x["close"].shift(1) <= vwap.shift(1))
            vol_ok = (x["rv_24"] < x["rv_96"]) if {"rv_24", "rv_96"}.issubset(x.columns) else pd.Series(True, index=x.index)
            sig = np.where(cross_up & (x["mom_6"] > 0) & vol_ok.fillna(False), 1, 0)
        elif mode == "vwap_pullback":
            sig = np.where((x["close"] > vwap) & (x["close_in_bar"] < 0.4) & (x["close"] > x["ema_200"]), 1,
                           np.where((x["close"] < vwap) & (x["close"] < x["ema_200"]), -1, 0))
        elif mode == "vol_spike_confirm":
            sig = np.where((x["rel_volume"] > 1.5) & (x["mom_3"] > 0), 1, np.where((x["rel_volume"] > 1.5) & (x["mom_3"] < 0), -1, 0))
        elif mode == "profile_node_proxy":
            node = x["close"].rolling(120).median()
            sig = np.where((x["close"] > node) & (x["mom_12"] > 0), 1,
                           np.where((x["close"] < node) & (x["mom_12"] < 0), -1, 0))
        else:  # obv-like
            obv = (np.sign(x["close"].diff()).fillna(0) * x["volume"]).cumsum()
            sig = np.where((obv > obv.rolling(20).mean()) & (x["mom_24"] > 0), 1,
                           np.where((obv < obv.rolling(20).mean()) & (x["mom_24"] < 0), -1, 0))

        return _finalize(x, pd.Series(sig, index=x.index))


class VolumeVwapCrossV1(VolumeVwapBaseStrategy): name, version, default_params = "volume_vwap_cross", "1.1", {"mode": "vwap_cross"}
class VolumeVwapCrossEventV1(VolumeVwapBaseStrategy): name, version, default_params = "volume_vwap_cross_event", "1.0", {"mode": "vwap_cross_event"}
class VolumeVwapCrossEventVolV1(VolumeVwapBaseStrategy): name, version, default_params = "volume_vwap_cross_event_vol", "1.0", {"mode": "vwap_cross_event_vol"}
class VolumeVwapPullbackV1(VolumeVwapBaseStrategy): name, version, default_params = "volume_vwap_pullback", "1.1", {"mode": "vwap_pullback"}
class VolumeSpikeConfirmV1(VolumeVwapBaseStrategy): name, version, default_params = "volume_spike_confirm", "1.1", {"mode": "vol_spike_confirm"}
class VolumeProfileNodeProxyV1(VolumeVwapBaseStrategy): name, version, default_params = "volume_profile_node_proxy", "1.1", {"mode": "profile_node_proxy"}
class VolumeObvLikeV1(VolumeVwapBaseStrategy): name, version, default_params = "volume_obv_like", "1.1", {"mode": "obv_like"}


# =========================
# 6) Multi-timeframe confirmation (5)
# =========================
class MTFBaseStrategy(BaseStrategy):
    family = "multi_timeframe"

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        x = self.prepare_dataset(df, ctx)
        mode = self.params.get("mode", "ema_htf")

        # proxy HTF from rolling windows on same timeframe
        htf_trend = x["close"].ewm(span=200, adjust=False).mean()
        ltf_trend = x["close"].ewm(span=50, adjust=False).mean()

        if mode == "ema_htf":
            sig = np.where((x["close"] > htf_trend) & (x["close"] > ltf_trend), 1, -1)
        elif mode == "mom_align":
            sig = np.where((x["mom_6"] > 0) & (x["mom_24"] > 0) & (x["mom_48"] > 0), 1, -1)
        elif mode == "rsi_trend":
            sig = np.where((x["rsi_14"] > 50) & (x["close"] > htf_trend), 1, np.where((x["rsi_14"] < 45) & (x["close"] < htf_trend), -1, 0))
        elif mode == "breakout_trend":
            sig = np.where((x["breakout_up"] == 1) & (x["close"] > htf_trend), 1, np.where((x["breakout_dn"] == 1) & (x["close"] < htf_trend), -1, 0))
        else:  # volatility trend filter
            sig = np.where((x["atr_pct"] < x["atr_pct"].rolling(60).mean()) & (x["close"] > htf_trend), 1, -1)

        return _finalize(x, pd.Series(sig, index=x.index))


class MtfEmaConfirmV1(MTFBaseStrategy): name, version, default_params = "mtf_ema_confirm", "1.1", {"mode": "ema_htf"}
class MtfMomentumAlignV1(MTFBaseStrategy): name, version, default_params = "mtf_momentum_align", "1.1", {"mode": "mom_align"}
class MtfRsiTrendV1(MTFBaseStrategy): name, version, default_params = "mtf_rsi_trend", "1.1", {"mode": "rsi_trend"}
class MtfBreakoutTrendV1(MTFBaseStrategy): name, version, default_params = "mtf_breakout_trend", "1.1", {"mode": "breakout_trend"}
class MtfVolTrendFilterV1(MTFBaseStrategy): name, version, default_params = "mtf_vol_trend_filter", "1.1", {"mode": "vol_trend"}


# =========================
# 7) Hybrid (5)
# =========================
class HybridBaseStrategy(BaseStrategy):
    family = "hybrid"

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        x = self.prepare_dataset(df, ctx)
        mode = self.params.get("mode", "trend_rsi")

        if mode == "trend_rsi":
            sig = np.where((x["close"] > x["ema_50"]) & (x["rsi_14"] > 52), 1, np.where((x["close"] < x["ema_50"]) & (x["rsi_14"] < 48), -1, 0))
        elif mode == "breakout_volume":
            sig = np.where((x["breakout_up"] == 1) & (x["rel_volume"] > 1.2), 1, np.where((x["breakout_dn"] == 1) & (x["rel_volume"] > 1.2), -1, 0))
        elif mode == "meanrev_regime":
            sig = np.where((x["regime"] != "bear") & (x["rsi_14"] < 35), 1, np.where((x["regime"] == "bear") & (x["rsi_14"] > 65), -1, 0))
        elif mode == "keltner_rsi":
            mid = x["ema_20"]; rng = 1.5 * x["atr_14"]
            sig = np.where((x["close"] < mid - rng) & (x["rsi_14"] < 40), 1, np.where((x["close"] > mid + rng) & (x["rsi_14"] > 60), -1, 0))
        else:  # momentum_vol
            sig = np.where((x["mom_12"] > 0) & (x["rv_24"] < x["rv_96"]), 1, np.where((x["mom_12"] < 0) & (x["rv_24"] > x["rv_96"]), -1, 0))

        return _finalize(x, pd.Series(sig, index=x.index))


class HybridTrendRsiV1(HybridBaseStrategy): name, version, default_params = "hybrid_trend_rsi", "1.1", {"mode": "trend_rsi"}
class HybridBreakoutVolumeV1(HybridBaseStrategy): name, version, default_params = "hybrid_breakout_volume", "1.1", {"mode": "breakout_volume"}
class HybridMeanRevRegimeV1(HybridBaseStrategy): name, version, default_params = "hybrid_meanrev_regime", "1.1", {"mode": "meanrev_regime"}
class HybridKeltnerRsiV1(HybridBaseStrategy): name, version, default_params = "hybrid_keltner_rsi", "1.1", {"mode": "keltner_rsi"}
class HybridMomentumVolV1(HybridBaseStrategy): name, version, default_params = "hybrid_momentum_vol", "1.1", {"mode": "momentum_vol"}


# =========================
# 8) Advanced (5)
# =========================
class AdvancedBaseStrategy(BaseStrategy):
    family = "advanced"

    def generate_signals(self, df: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        x = self.prepare_dataset(df, ctx)
        mode = self.params.get("mode", "consensus_trend")

        if mode == "consensus_trend":
            trend_ok = (x["close"] > x["ema_50"]) & (x["mom_6"] > 0) & (x["mom_24"] > 0)
            trend_bad = (x["close"] < x["ema_50"]) & (x["mom_6"] < 0) & (x["mom_24"] < 0)
            sig = np.where(trend_ok & (x["rsi_14"] > 50), 1, np.where(trend_bad & (x["rsi_14"] < 50), -1, 0))
        elif mode == "meta_proxy":
            score = 0.4 * x["mom_12"].fillna(0) - 0.2 * x["atr_pct"].fillna(0) + 0.2 * x["z_rel_volume"].fillna(0) + 0.2 * x["dist_ema_50"].fillna(0)
            sig = np.where(score > score.rolling(80).quantile(0.6), 1, np.where(score < score.rolling(80).quantile(0.4), -1, 0))
        elif mode == "breakout_bias":
            up_break = (x["close"] > x["hh_20"]) & (x["close"] > x["ema_20"]) & (x["rel_volume"] > 1.1)
            dn_break = (x["close"] < x["ll_20"]) & (x["close"] < x["ema_20"]) & (x["rel_volume"] > 1.1)
            sig = np.where(up_break, 1, np.where(dn_break, -1, 0))
        elif mode == "regime_transition":
            up = x.get("p_stay_bull", pd.Series(0.33, index=x.index))
            dn = x.get("p_stay_bear", pd.Series(0.33, index=x.index))
            sig = np.where((up > 0.60) & (x["mom_6"] > 0), 1, np.where((dn > 0.60) & (x["mom_6"] < 0), -1, 0))
        else:  # ensemble_vote
            v1 = (x["mom_6"] > 0).astype(int) - (x["mom_6"] < 0).astype(int)
            v2 = (x["rsi_14"] > 50).astype(int) - (x["rsi_14"] < 50).astype(int)
            v3 = (x["close"] > x["ema_50"]).astype(int) - (x["close"] < x["ema_50"]).astype(int)
            vote = v1 + v2 + v3
            sig = np.where(vote >= 2, 1, np.where(vote <= -2, -1, 0))

        return _finalize(x, pd.Series(sig, index=x.index))


class AdvancedLabelAgreeV1(AdvancedBaseStrategy): name, version, default_params = "advanced_label_agree", "1.1", {"mode": "consensus_trend"}
class AdvancedMetaProxyV1(AdvancedBaseStrategy): name, version, default_params = "advanced_meta_proxy", "1.1", {"mode": "meta_proxy"}
class AdvancedTripleBarrierBiasV1(AdvancedBaseStrategy): name, version, default_params = "advanced_tb_bias", "1.1", {"mode": "breakout_bias"}
class AdvancedRegimeTransitionV1(AdvancedBaseStrategy): name, version, default_params = "advanced_regime_transition", "1.1", {"mode": "regime_transition"}
class AdvancedEnsembleVoteV1(AdvancedBaseStrategy): name, version, default_params = "advanced_ensemble_vote", "1.1", {"mode": "ensemble_vote"}


ALL_STRATEGIES = [
    MomentumEMACrossV1, MomentumEMACrossEventV1, MomentumEMACrossEventVolV1, MomentumEMACrossEventVolDistV1, MomentumEMACrossEventVolSlopeV1, MomentumMACDV1, MomentumTrixV1, MomentumROCV1, MomentumDualV1,
    MeanRevRSIV1, MeanRevBollingerV1, MeanRevStochV1, MeanRevZScoreV1, MeanRevDrawdownV1, MeanRevDrawdownLooseV1, MeanRevDrawdownDeepV1, MeanRevDrawdownLoose05V1, MeanRevDrawdownLooseVolV1, MeanRevDrawdownLoose05FastExitV1, MeanRevDrawdownLoose05SlowExitV1,
    BreakoutDonchianV1, BreakoutPivotV1, BreakoutRangeV1, BreakoutSwingV1, BreakoutOrderBlockProxyV1,
    VolAtrExpandV1, VolKeltnerV1, VolRvRegimeV1, VolTrendRegimeV1, VolCompressionBreakV1,
    VolumeVwapCrossV1, VolumeVwapCrossEventV1, VolumeVwapCrossEventVolV1, VolumeVwapPullbackV1, VolumeSpikeConfirmV1, VolumeProfileNodeProxyV1, VolumeObvLikeV1,
    MtfEmaConfirmV1, MtfMomentumAlignV1, MtfRsiTrendV1, MtfBreakoutTrendV1, MtfVolTrendFilterV1,
    HybridTrendRsiV1, HybridBreakoutVolumeV1, HybridMeanRevRegimeV1, HybridKeltnerRsiV1, HybridMomentumVolV1,
    AdvancedLabelAgreeV1, AdvancedMetaProxyV1, AdvancedTripleBarrierBiasV1, AdvancedRegimeTransitionV1, AdvancedEnsembleVoteV1,
]


def register_solusdt_strategies() -> None:
    for cls in ALL_STRATEGIES:
        spec = StrategySpec(
            name=cls.name,
            version=cls.version,
            family=cls.family,
            strategy_cls=cls,
            default_params=cls.default_params,
        )
        try:
            registry.register(spec)
        except ValueError as e:
            # idempotent in repeated imports only
            if "already registered" in str(e):
                continue
            raise


register_solusdt_strategies()
