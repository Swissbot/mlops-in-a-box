from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI
from .schemas import PredictRequest, PredictResponse
from ..predict import predict_proba

app = FastAPI(title="mlops-in-a-box", version="0.1.0")

MODEL_PATH = Path("models/latest/model.joblib")

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run training first: `make train`.")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": MODEL_PATH.exists()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = load_model()
    X = np.asarray(req.X, dtype=np.float32)
    p = predict_proba(model, X)
    return PredictResponse(proba=[float(x) for x in p])

def main():
    # entry-point for completeness; typically use: uvicorn mlops_box.serving.app:app
    import uvicorn
    uvicorn.run("mlops_box.serving.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
