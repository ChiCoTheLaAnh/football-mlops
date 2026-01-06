from pathlib import Path
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/xg_baseline.pkl")
WINPROB_PATH = Path("models/winprob.pkl")
winprob = None

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
    global model, winprob

    # xG model (optional)
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None

    # WinProb model (optional)
    if WINPROB_PATH.exists():
        winprob = joblib.load(WINPROB_PATH)
    else:
        winprob = None

@app.get("/")
def root():
    return {
        "service": "football-mlops-api",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "xg_loaded": model is not None,
        "winprob_loaded": winprob is not None
    }


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

class WinProbState(BaseModel):
    minute: float = Field(..., ge=0, le=130)
    score_diff: float
    xg_diff: float
    home_red: int = 0
    away_red: int = 0

class WinProbRequest(BaseModel):
    states: list[WinProbState]

@app.post("/predict/win-prob")
def predict_win_prob(req: WinProbRequest):
    if winprob is None:
        raise HTTPException(status_code=500, detail="WinProb model not loaded")

    pack = winprob
    model_wp = pack["model"]
    classes = pack["classes"]

    df = pd.DataFrame([s.model_dump() for s in req.states])
    X = df[["minute", "score_diff", "xg_diff", "home_red", "away_red"]]

    proba = model_wp.predict_proba(X)
    results = [{cls: float(p[i]) for i, cls in enumerate(classes)} for p in proba]

    return {"classes": classes, "results": results}

from src.api.state_store import STORE, MatchState

class StreamInitRequest(BaseModel):
    match_id: str
    home_team: str
    away_team: str

@app.post("/stream/init")
def stream_init(req: StreamInitRequest):
    STORE[req.match_id] = MatchState(home_team=req.home_team, away_team=req.away_team)
    return {"ok": True, "match_id": req.match_id, "state": STORE[req.match_id].to_dict()}

class StreamEvent(BaseModel):
    match_id: str
    minute: float = Field(..., ge=0, le=130)
    # minimal event types for demo
    type: str = Field(..., description="shot | goal | red")
    team: str = Field(..., description="home | away")
    xg: float = 0.0  # only for shot events

@app.post("/stream/event")
def stream_event(ev: StreamEvent):
    if ev.match_id not in STORE:
        raise HTTPException(status_code=404, detail="Unknown match_id. Call /stream/init first.")
    st = STORE[ev.match_id]

    st.minute = float(ev.minute)

    if ev.type == "shot":
        if ev.team == "home":
            st.home_xg += float(ev.xg)
        elif ev.team == "away":
            st.away_xg += float(ev.xg)
        else:
            raise HTTPException(status_code=400, detail="team must be home|away")
    elif ev.type == "goal":
        if ev.team == "home":
            st.home_goals += 1
        elif ev.team == "away":
            st.away_goals += 1
        else:
            raise HTTPException(status_code=400, detail="team must be home|away")
    elif ev.type == "red":
        if ev.team == "home":
            st.home_red += 1
        elif ev.team == "away":
            st.away_red += 1
        else:
            raise HTTPException(status_code=400, detail="team must be home|away")
    else:
        raise HTTPException(status_code=400, detail="type must be shot|goal|red")

    # Return win probabilities for current state
    if winprob is None:
        return {"state": st.to_dict(), "winprob": None}

    df = pd.DataFrame([st.to_features()])
    X = df[["minute", "score_diff", "xg_diff", "home_red", "away_red"]]
    proba = winprob["model"].predict_proba(X)[0]
    classes = winprob["classes"]
    wp = {cls: float(proba[i]) for i, cls in enumerate(classes)}

    return {"state": st.to_dict(), "winprob": wp}

@app.get("/stream/state/{match_id}")
def stream_state(match_id: str):
    if match_id not in STORE:
        raise HTTPException(status_code=404, detail="Unknown match_id")
    st = STORE[match_id]
    return {"match_id": match_id, "state": st.to_dict()}
