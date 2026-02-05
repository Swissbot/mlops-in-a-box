from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

@dataclass
class Metrics:
    accuracy: float
    f1: float
    auc: float

def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Metrics:
    y_pred = (proba >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    # AUC requires both classes present; guard for tiny splits
    try:
        auc = float(roc_auc_score(y_true, proba))
    except Exception:
        auc = float("nan")
    return Metrics(accuracy=acc, f1=f1, auc=auc)
