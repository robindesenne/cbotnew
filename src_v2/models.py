from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_meta_model(Xtr, ytr, Xcal, ycal):
    cands = {
        "logreg": make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=10000, solver="saga", C=0.5, class_weight="balanced", random_state=42)),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.04, max_iter=500, random_state=42),
    }
    best_name, best_score, best_model = None, -1, None
    for name, m in cands.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xcal)[:, 1]
        # precision-oriented simple score
        score = float(np.nanmean((p[ycal.values == 1]) if (ycal == 1).any() else [0]))
        if score > best_score:
            best_name, best_score, best_model = name, score, m
    cal = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    cal.fit(Xcal, ycal)
    return best_name, cal
