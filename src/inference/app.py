import time
import pandas as pd
from fastapi import FastAPI

from src.features.geometry import shot_distance, shot_angle
from src.inference.schemas import ShotEvent, PredictionResponse
from src.inference.model_loader import load_model, DEFAULT_MODEL_PATH
from src.inference.logging import log_event

app = FastAPI(title="xG Real-Time Prediction Service", version="1.0.0")

MODEL = load_model(DEFAULT_MODEL_PATH)
MODEL_VERSION = DEFAULT_MODEL_PATH.name

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=PredictionResponse)
def predict(evt: ShotEvent):
    t0 = time.perf_counter()

    distance = shot_distance(evt.x, evt.y)
    angle = shot_angle(evt.x, evt.y)

    X = pd.DataFrame([{
        "distance": distance,
        "angle": angle,
        "shot_body_part": evt.shot_body_part,
        "shot_type": evt.shot_type,
        "play_pattern": evt.play_pattern,
    }])

    xg = float(MODEL.predict_proba(X)[:, 1][0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    log_event({
        "input": evt.model_dump(),
        "features": {"distance": distance, "angle": angle},
        "xg": xg,
        "latency_ms": latency_ms,
        "model_version": MODEL_VERSION,
    })

    return PredictionResponse(xg=xg, model_version=MODEL_VERSION, latency_ms=latency_ms)
