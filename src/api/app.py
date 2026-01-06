from pathlib import Path
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/xg_baseline.pkl")

app = FastAPI(title="Football Match Intelligence API", version="0.1.0")

model = None

class Shot(BaseModel):
    distance_to_goal: float = Field(..., ge=0)
    shot_angle: float = Field(..., ge=0)
    body_part: str
    shot_type: str

class XGRequest(BaseModel):
    shots: List[Shot]

class XGResponseItem(BaseModel):
    xg: float

class XGResponse(BaseModel):
    results: List[XGResponseItem]

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing model file: {MODEL_PATH}. Run training to create models/xg_baseline.pkl"
        )
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict/xg", response_model=XGResponse)
def predict_xg(req: XGRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([s.model_dump() for s in req.shots])

    # Ensure columns order/names match training
    expected = ["distance_to_goal", "shot_angle", "shot_type", "body_part"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    X = df[expected]
    proba = model.predict_proba(X)[:, 1]
    return {"results": [{"xg": float(p)} for p in proba]}
