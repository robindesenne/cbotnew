from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier


def train_calibrated_model(Xtr, ytr, Xcal, ycal):
    model = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=350, random_state=42)
    model.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(Xcal, ycal)
    return cal


def choose_threshold_from_cal(p_cal, y_cal, min_precision: float = 0.53, floor: float = 0.55, cap: float = 0.90) -> float:
    y = np.asarray(y_cal).astype(int)
    best_t, best_f1 = floor, -1.0
    for t in np.linspace(floor, cap, 36):
        pred = (p_cal >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if prec < min_precision:
            continue
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return float(best_t)
