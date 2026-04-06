from __future__ import annotations

GLOBAL_CONFIG = {
    "pairs": ["SOLUSDT"],
    "timeframes": ["15m", "1h", "4h"],
    "label_horizon_bars": 24,
    "embargo_bars": 2,
    "outer_splits": 4,
    "inner_splits": 3,
    "min_train_rows": 400,
    "min_valid_rows": 120,
    "min_test_rows": 120,
    "threshold_grid": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "min_precision": 0.52,
    "max_ml_rows_per_strategy": 12000,
    "solusdt": {
        "symbol": "SOLUSDT",
        "default_timeframe": "1h",
        "supported_timeframes": ["15m", "1h", "4h"],
        "history_start": "2021-01-01",
        "history_end": "2100-01-01",
        "market_type": "spot",
        "prefer_local": True,
        "cache_dir": "data/cache/solusdt",
        "feature_profile": "ultra_benchmark_v1",
    },
}
