import math

import numpy as np
import pytest

from mfp.features import FEATURE_NAMES, FeatureBuilder, mechanical_power_w


def test_hand_computed_row():
    row = np.array([["M", 300.0, 310.0, 1500, 40.0, 100]], dtype=object)
    out = FeatureBuilder().fit(row).transform(row)[0]
    feats = dict(zip(FEATURE_NAMES, out))
    assert (feats["type_L"], feats["type_M"], feats["type_H"]) == (0, 1, 0)
    assert feats["temperature_diff_k"] == pytest.approx(10.0)
    assert feats["power_w"] == pytest.approx(40 * 1500 * 2 * math.pi / 60)
    assert feats["wear_x_torque"] == pytest.approx(4000.0)


def test_output_width_matches_names():
    rows = np.array([["L", 298, 308, 1400, 45, 10], ["H", 301, 311, 1600, 30, 200]], dtype=object)
    out = FeatureBuilder().fit_transform(rows)
    assert out.shape == (2, len(FEATURE_NAMES))
    assert list(FeatureBuilder().get_feature_names_out()) == FEATURE_NAMES


def test_power_in_watts():
    assert mechanical_power_w(10, 60) == pytest.approx(2 * math.pi * 10)


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="unknown product type"):
        FeatureBuilder().fit_transform(np.array([["Z", 300, 310, 1500, 40, 100]], dtype=object))


def test_wrong_number_of_columns_is_rejected():
    with pytest.raises(ValueError, match="expected 6 columns"):
        FeatureBuilder().fit_transform(np.array([["L", 300, 310]], dtype=object))
