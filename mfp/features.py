"""Feature engineering as a scikit-learn transformer, so training and serving share one code path.

The engineered features follow the failure rules described in the dataset's documentation:
  * heat dissipation failure depends on the gap between process and air temperature;
  * power failure depends on mechanical power (torque x angular speed);
  * overstrain failure depends on tool wear x torque;
  * tool wear failure depends on tool wear itself.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

PRODUCT_TYPES = ("L", "M", "H")
FEATURE_NAMES = [
    "type_L",
    "type_M",
    "type_H",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temperature_diff_k",
    "power_w",
    "wear_x_torque",
]


def mechanical_power_w(torque_nm, rotational_speed_rpm):
    """Power in watts: torque (Nm) x angular speed (rad/s)."""
    return np.asarray(torque_nm, dtype=float) * np.asarray(rotational_speed_rpm, dtype=float) * 2 * math.pi / 60


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Turns raw rows [type, air K, process K, rpm, torque Nm, tool wear min] into numeric features."""

    def fit(self, X, y=None):
        self._check(X)
        self.n_features_in_ = 6
        return self

    def transform(self, X):
        X = self._check(X)
        types = X[:, 0].astype(str)
        sensors = X[:, 1:].astype(float)
        air, proc, rpm, torque, wear = sensors.T
        one_hot = np.stack([(types == t).astype(float) for t in PRODUCT_TYPES], axis=1)
        engineered = np.stack(
            [proc - air, mechanical_power_w(torque, rpm), wear * torque],
            axis=1,
        )
        return np.hstack([one_hot, sensors, engineered])

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES, dtype=object)

    @staticmethod
    def _check(X):
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != 6:
            raise ValueError(f"expected 6 columns [type + 5 sensors], got shape {X.shape}")
        unknown = set(X[:, 0].astype(str)) - set(PRODUCT_TYPES)
        if unknown:
            raise ValueError(f"unknown product type(s): {sorted(unknown)}; expected one of {PRODUCT_TYPES}")
        return X
