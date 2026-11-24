import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.metrics import roc_auc_score  # noqa: E402

from mfp.models import make_model  # noqa: E402
from mfp.nn import TorchMLPClassifier  # noqa: E402


def _blobs(n=600, pos_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    X = rng.normal(size=(n, 4)) + y[:, None] * 2.5
    return X, y


def test_learns_a_separable_problem():
    X, y = _blobs()
    clf = TorchMLPClassifier(epochs=40, hidden=(16,)).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)
    assert roc_auc_score(y, proba[:, 1]) > 0.95


def test_same_seed_gives_same_model():
    X, y = _blobs()
    a = TorchMLPClassifier(epochs=5, random_state=7).fit(X, y).predict_proba(X)
    b = TorchMLPClassifier(epochs=5, random_state=7).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)


def test_runs_inside_the_shared_pipeline():
    from mfp.data import AI4IDataset

    X, y = AI4IDataset().features_and_target()
    model = make_model("pytorch_mlp", fast=True).fit(X[:2000], y[:2000])
    p = model.predict_proba(X[2000:2100])[:, 1]
    assert p.shape == (100,) and np.all((p >= 0) & (p <= 1))
