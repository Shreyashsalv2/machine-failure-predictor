"""Load, validate and clean the AI4I 2020 predictive maintenance dataset with pandas."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import PRODUCT_TYPES

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"

# Column names in the original CSV -> short names used in code.
RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}
SENSOR_COLUMNS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
RAW_FEATURES = ["type"] + SENSOR_COLUMNS
TARGET = "machine_failure"
FAILURE_MODES = ["twf", "hdf", "pwf", "osf", "rnf"]
# Dropped before modelling: IDs carry no signal, and the five failure-mode flags
# are only known after a failure, so using them would leak the answer.
LEAKY_OR_ID = ["udi", "product_id"] + FAILURE_MODES


class DataValidationError(ValueError):
    """Raised when the CSV does not look like the AI4I 2020 dataset."""


@dataclass
class DataReport:
    rows: int
    failures: int
    failure_rate: float
    type_counts: dict
    missing_values: int
    duplicate_rows: int
    failure_without_mode: int
    mode_without_failure: int
    notes: list = field(default_factory=list)


class AI4IDataset:
    """Reads the CSV, checks its schema, and returns model-ready features and labels."""

    def __init__(self, path: str | Path = DATA_PATH):
        self.path = Path(path)
        self._frame: pd.DataFrame | None = None

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = self.load()
        return self._frame

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, encoding="utf-8-sig")
        missing = [c for c in RENAME if c not in df.columns]
        if missing:
            raise DataValidationError(f"missing columns: {missing}")
        df = df.rename(columns=RENAME)
        bad_types = set(df["type"].unique()) - set(PRODUCT_TYPES)
        if bad_types:
            raise DataValidationError(f"unknown product types: {sorted(bad_types)}")
        for col in SENSOR_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        df[TARGET] = df[TARGET].astype(int)
        return df

    def report(self) -> DataReport:
        df = self.frame
        any_mode = df[FAILURE_MODES].sum(axis=1) > 0
        failure = df[TARGET] == 1
        rep = DataReport(
            rows=len(df),
            failures=int(failure.sum()),
            failure_rate=round(float(failure.mean()), 4),
            type_counts={k: int(v) for k, v in df["type"].value_counts().items()},
            missing_values=int(df.isna().sum().sum()),
            duplicate_rows=int(df.drop(columns=["udi", "product_id"]).duplicated().sum()),
            failure_without_mode=int((failure & ~any_mode).sum()),
            mode_without_failure=int((~failure & any_mode).sum()),
        )
        if rep.failure_without_mode:
            rep.notes.append(f"{rep.failure_without_mode} rows are failures with no failure mode flagged; kept as failures.")
        if rep.mode_without_failure:
            modes = df.loc[~failure & any_mode, FAILURE_MODES].sum()
            which = ", ".join(f"{int(v)} {k.upper()}" for k, v in modes.items() if v)
            rep.notes.append(
                f"{rep.mode_without_failure} rows flag a failure mode ({which}) but are labelled as not failing; "
                "the label used is the 'Machine failure' column."
            )
        return rep

    def features_and_target(self) -> tuple[np.ndarray, np.ndarray]:
        """Raw feature matrix (type as a string column, then the five sensors) and the 0/1 target."""
        df = self.frame.drop(columns=LEAKY_OR_ID)
        X = df[RAW_FEATURES].to_numpy(dtype=object)
        y = df[TARGET].to_numpy(dtype=int)
        return X, y
