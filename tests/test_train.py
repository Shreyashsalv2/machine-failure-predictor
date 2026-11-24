import json

import joblib

from mfp.train import run


def test_training_smoke_run(tmp_path):
    metrics = run(tmp_path, None, fast=True, models=["logistic_regression", "gradient_boosting"])
    assert (tmp_path / "model.joblib").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["served_model"] in {"logistic_regression", "gradient_boosting"}
    assert saved["split"]["test_rows"] == 2000 and saved["split"]["train_rows"] == 8000
    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["name"] == metrics["served_model"]
    for m in saved["models"].values():
        assert 0 <= m["test"]["pr_auc"] <= 1
