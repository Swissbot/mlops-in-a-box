from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    # expects row-major features
    X: List[List[float]] = Field(..., description="Batch of feature rows")

class PredictResponse(BaseModel):
    proba: List[float]
