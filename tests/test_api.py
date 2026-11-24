import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
VALID = {"type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
         "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_valid_reading():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"failure_probability", "predicted_failure", "threshold", "model", "derived", "warnings"}
    assert 0 <= body["failure_probability"] <= 1


@pytest.mark.parametrize("change", [
    {"type": "X"},
    {"torque_nm": -1},
    {"rotational_speed_rpm": 99999},
    {"unexpected": 1},
])
def test_predict_rejects_bad_input(change):
    assert client.post("/predict", json={**VALID, **change}).status_code == 422


def test_predict_rejects_missing_field():
    body = dict(VALID)
    body.pop("tool_wear_min")
    assert client.post("/predict", json=body).status_code == 422


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert len(r.json()["comparison"]) == 4


def test_demo_page_and_figures():
    page = client.get("/")
    assert page.status_code == 200 and "Machine Failure Predictor" in page.text
    fig = client.get("/figures/pr_curves.png")
    assert fig.status_code == 200 and fig.headers["content-type"] == "image/png"


def test_app_imports_without_public_folder(tmp_path):
    """Vercel serves public/ from its CDN and leaves it out of the function bundle; the API must still start."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    shutil.copytree(root / "mfp", tmp_path / "mfp")
    shutil.copytree(root / "artifacts", tmp_path / "artifacts")
    shutil.copy(root / "app.py", tmp_path / "app.py")
    code = ("from fastapi.testclient import TestClient; import app; c = TestClient(app.app); "
            "assert c.get('/health').status_code == 200; "
            "assert c.post('/predict', json=%r).status_code == 200" % VALID)
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
