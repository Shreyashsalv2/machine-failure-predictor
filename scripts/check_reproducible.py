"""Retrain from scratch and check the scores match the committed artifacts/metrics.json.

Used in CI so the numbers in the README cannot silently drift from what the code produces.
scikit-learn runs are deterministic; the PyTorch network may differ slightly across CPUs, hence the tolerance.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mfp.train import run  # noqa: E402

TOLERANCE = {"pytorch_mlp": 0.03}
DEFAULT_TOL = 0.005


def main():
    committed = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        fresh = run(Path(tmp), None)
    problems = []
    if fresh["served_model"] != committed["served_model"]:
        problems.append(f"served model changed: {committed['served_model']} -> {fresh['served_model']}")
    for name, old in committed["models"].items():
        new = fresh["models"][name]
        tol = TOLERANCE.get(name, DEFAULT_TOL)
        for key, a, b in [("cv_pr_auc_mean", old["cv_pr_auc_mean"], new["cv_pr_auc_mean"]),
                          ("test_pr_auc", old["test"]["pr_auc"], new["test"]["pr_auc"])]:
            if abs(a - b) > tol:
                problems.append(f"{name} {key}: committed {a} vs fresh {b} (tolerance {tol})")
    if problems:
        print("NOT REPRODUCIBLE:\n- " + "\n- ".join(problems))
        sys.exit(1)
    print("Reproducible: fresh run matches artifacts/metrics.json")


if __name__ == "__main__":
    main()
