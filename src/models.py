from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


@dataclass
class ModelResult:
    name: str
    model: object
    auc: float


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.15):
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + valid_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def train_candidates(X_train, y_train, X_valid, y_valid) -> list[ModelResult]:
    candidates = {
        "logreg": make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=10000, solver="saga", C=0.5, class_weight="balanced", random_state=42)),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42),
    }

    out = []
    for name, m in candidates.items():
        m.fit(X_train, y_train)
        p = m.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, p) if len(np.unique(y_valid)) > 1 else 0.5
        out.append(ModelResult(name=name, model=m, auc=float(auc)))
    return out


def calibrate_model(model, X_valid, y_valid):
    cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(X_valid, y_valid)
    return cal


def choose_threshold(proba: pd.Series, y: pd.Series, min_precision: float = 0.52) -> float:
    best_t, best_f1 = 0.6, -1
    for t in np.linspace(0.5, 0.9, 41):
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        if prec < min_precision:
            continue
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)
