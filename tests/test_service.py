import pytest

from mfp.models import MODEL_LABELS
from mfp.service import ModelService

NORMAL = ("L", 300.0, 310.0, 1500, 40.0, 100)
OVERSTRAIN = ("L", 300.0, 310.0, 1350, 62.0, 210)  # high wear x torque, the overstrain rule


@pytest.fixture(scope="module")
def svc():
    return ModelService()


def test_loads_the_saved_model(svc):
    assert svc.name in MODEL_LABELS and svc.name != "pytorch_mlp"
    assert 0 < svc.threshold < 1


def test_probability_is_valid(svc):
    r = svc.predict(*NORMAL)
    assert 0 <= r["failure_probability"] <= 1
    assert r["predicted_failure"] is (r["failure_probability"] >= r["threshold"])
    assert r["warnings"] == []


def test_overstrain_scores_higher_than_normal(svc):
    assert svc.predict(*OVERSTRAIN)["failure_probability"] > svc.predict(*NORMAL)["failure_probability"]


def test_out_of_range_input_gets_a_warning(svc):
    r = svc.predict("L", 300.0, 310.0, 1500, 120.0, 100)
    assert len(r["warnings"]) == 1 and "torque_nm" in r["warnings"][0]


def test_info_lists_every_model(svc):
    info = svc.info()
    assert {m["model"] for m in info["comparison"]} == set(MODEL_LABELS)
    assert info["dataset"]["rows"] == 10_000
