"""The candidate models. Every one is a Pipeline that starts from the same FeatureBuilder."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FeatureBuilder

SEED = 42

MODEL_LABELS = {
    "logistic_regression": "Logistic regression (baseline)",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "pytorch_mlp": "PyTorch neural network (MLP)",
}


def make_model(name: str, fast: bool = False) -> Pipeline:
    """Build a fresh, unfitted pipeline. `fast=True` shrinks the models for quick tests."""
    if name == "logistic_regression":
        steps = [("scale", StandardScaler()),
                 ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]
    elif name == "random_forest":
        steps = [("clf", RandomForestClassifier(
            n_estimators=60 if fast else 400, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))]
    elif name == "gradient_boosting":
        steps = [("clf", HistGradientBoostingClassifier(
            max_iter=60 if fast else 300, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=SEED))]
    elif name == "pytorch_mlp":
        from .nn import TorchMLPClassifier

        steps = [("scale", StandardScaler()),
                 ("clf", TorchMLPClassifier(epochs=8 if fast else 150, random_state=SEED))]
    else:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_LABELS)}")
    return Pipeline([("features", FeatureBuilder())] + steps)


def available_models() -> list[str]:
    """All models, minus the PyTorch one when torch is not installed."""
    names = list(MODEL_LABELS)
    try:
        import torch  # noqa: F401
    except ImportError:
        names.remove("pytorch_mlp")
    return names
