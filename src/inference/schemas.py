from pydantic import BaseModel, Field

class ShotEvent(BaseModel):
    x: float = Field(..., ge=0, le=120)
    y: float = Field(..., ge=0, le=80)
    shot_body_part: str
    shot_type: str
    play_pattern: str

class PredictionResponse(BaseModel):
    xg: float
    model_version: str
    latency_ms: float
