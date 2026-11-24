"""FastAPI service. Run locally with `uvicorn app:app --reload`; Vercel and Docker use the same `app`."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from mfp import __version__
from mfp.service import get_service

PUBLIC = Path(__file__).resolve().parent / "public"

app = FastAPI(
    title="Machine Failure Predictor",
    version=__version__,
    description="Predicts machine failure from sensor readings (AI4I 2020 dataset).",
)


class SensorReading(BaseModel):
    """One reading. Bounds are physical sanity limits; values outside the training range still get a warning."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}]})

    type: Literal["L", "M", "H"] = Field(description="Product quality variant: L (low), M (medium), H (high)")
    air_temperature_k: float = Field(ge=250, le=350, description="Air temperature in kelvin")
    process_temperature_k: float = Field(ge=250, le=380, description="Process temperature in kelvin")
    rotational_speed_rpm: float = Field(ge=0, le=5000, description="Spindle speed in rpm")
    torque_nm: float = Field(ge=0, le=150, description="Torque in newton-metres")
    tool_wear_min: float = Field(ge=0, le=400, description="Tool wear in minutes")


class Prediction(BaseModel):
    failure_probability: float
    predicted_failure: bool
    threshold: float
    model: str
    derived: dict
    warnings: list[str]


@app.get("/health")
def health():
    svc = get_service()
    return {"status": "ok", "model": svc.name, "version": __version__}


@app.get("/model-info")
def model_info():
    return get_service().info()


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading):
    return get_service().predict(
        product_type=reading.type,
        air_temperature_k=reading.air_temperature_k,
        process_temperature_k=reading.process_temperature_k,
        rotational_speed_rpm=reading.rotational_speed_rpm,
        torque_nm=reading.torque_nm,
        tool_wear_min=reading.tool_wear_min,
    )


# On Vercel, public/ is served by the CDN and is not bundled into the Python function,
# so the page and figures are only mounted here when the folder exists (local runs and Docker).
if (PUBLIC / "index.html").is_file():
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(PUBLIC / "index.html")

if (PUBLIC / "figures").is_dir():
    app.mount("/figures", StaticFiles(directory=PUBLIC / "figures"), name="figures")
