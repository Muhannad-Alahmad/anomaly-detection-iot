from pydantic import BaseModel, Field


class SensorEvent(BaseModel):
    timestamp: str = Field(..., description="ISO-8601 timestamp, e.g. 2025-01-01T12:00:00Z")
    sequence: int = Field(..., ge=1)
    station_id: str = Field(..., min_length=1, max_length=64)

    temperature_c: float = Field(..., ge=-50, le=200)
    humidity_pct: float = Field(..., ge=0, le=100)
    sound_db: float = Field(..., ge=0, le=200)
