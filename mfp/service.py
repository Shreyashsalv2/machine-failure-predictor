"""Loads the saved model once and turns validated sensor readings into a failure prediction."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .features import mechanical_power_w

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


class ModelService:
    def __init__(self, artifacts: str | Path = ARTIFACTS):
        artifacts = Path(artifacts)
        bundle = joblib.load(artifacts / "model.joblib")
        self.model = bundle["model"]
        self.name = bundle["name"]
        self.label = bundle["label"]
        self.threshold = float(bundle["threshold"])
        self.metrics = json.loads((artifacts / "metrics.json").read_text())

    def predict(self, product_type: str, air_temperature_k: float, process_temperature_k: float,
                rotational_speed_rpm: float, torque_nm: float, tool_wear_min: float) -> dict:
        row = np.array([[product_type, air_temperature_k, process_temperature_k,
                         rotational_speed_rpm, torque_nm, tool_wear_min]], dtype=object)
        p = float(self.model.predict_proba(row)[0, 1])
        values = {"air_temperature_k": air_temperature_k, "process_temperature_k": process_temperature_k,
                  "rotational_speed_rpm": rotational_speed_rpm, "torque_nm": torque_nm,
                  "tool_wear_min": tool_wear_min}
        warnings = [
            f"{col} = {v:g} is outside the training range {lo:g}–{hi:g}, so this prediction is an extrapolation."
            for col, v in values.items()
            for lo, hi in [self.metrics["training_ranges"][col]]
            if not lo <= v <= hi
        ]
        return {
            "failure_probability": round(p, 4),
            "predicted_failure": p >= self.threshold,
            "threshold": round(self.threshold, 4),
            "model": self.name,
            "derived": {
                "temperature_diff_k": round(process_temperature_k - air_temperature_k, 2),
                "power_w": round(float(mechanical_power_w(torque_nm, rotational_speed_rpm)), 1),
                "wear_x_torque": round(tool_wear_min * torque_nm, 1),
            },
            "warnings": warnings,
        }

    def info(self) -> dict:
        m = self.metrics
        return {
            "served_model": self.name,
            "served_model_label": self.label,
            "threshold": round(self.threshold, 4),
            "trained_at": m["trained_at"],
            "dataset": {k: m["dataset"][k] for k in ("name", "rows", "failures", "failure_rate", "sha256")},
            "split": m["split"],
            "training_ranges": m["training_ranges"],
            "comparison": [
                {"model": n, "label": r["label"], "cv_pr_auc_mean": r["cv_pr_auc_mean"],
                 "cv_pr_auc_std": r["cv_pr_auc_std"], "test": r["test"]}
                for n, r in m["models"].items()
            ],
            "permutation_importance": m.get("permutation_importance", {}),
        }


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    return ModelService()
