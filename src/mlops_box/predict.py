from __future__ import annotations
import numpy as np

def predict_proba(model, X: np.ndarray) -> np.ndarray:
    # sklearn convention: predict_proba returns [N,2]
    p = model.predict_proba(X)
    return p[:, 1].astype(float)
