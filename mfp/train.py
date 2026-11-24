"""Train and compare the models, then save the best scikit-learn model for the API.

    python -m mfp.train                 # full run: writes artifacts/ and public/figures/
    python -m mfp.train --fast --out /tmp/a --figures /tmp/f   # quick smoke run

Protocol (fixed seeds, so the run is reproducible):
  1. Hold out 20% of rows as a test set (stratified by the label).
  2. On the other 80%, run stratified 5-fold cross-validation for every model.
     PR-AUC (average precision) is the main score because only ~3% of rows are failures.
  3. Pick each model's decision threshold from its out-of-fold predictions (max F1),
     so the test set is never used to tune anything.
  4. Refit each model on the full 80% and score it once on the test set.
  5. Save the scikit-learn model with the best cross-validated PR-AUC for serving.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import __version__
from .data import AI4IDataset, RAW_FEATURES
from .evaluate import best_f1_threshold, summarize
from .features import FEATURE_NAMES
from .models import MODEL_LABELS, SEED, available_models, make_model

ROOT = Path(__file__).resolve().parent.parent


def cross_validate(model, X, y, n_splits=5):
    """Out-of-fold probabilities plus the PR-AUC of each fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    for train_idx, val_idx in skf.split(X, y):
        m = clone(model).fit(X[train_idx], y[train_idx])
        oof[val_idx] = m.predict_proba(X[val_idx])[:, 1]
        fold_scores.append(average_precision_score(y[val_idx], oof[val_idx]))
    return oof, fold_scores


def run(out_dir: Path, fig_dir: Path | None, fast: bool = False, models=None) -> dict:
    t0 = time.time()
    ds = AI4IDataset()
    report = ds.report()
    X, y = ds.features_and_target()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    names = models or available_models()
    results, fitted, test_proba = {}, {}, {}
    for name in names:
        t = time.time()
        model = make_model(name, fast=fast)
        oof, folds = cross_validate(model, X_tr, y_tr)
        threshold = best_f1_threshold(y_tr, oof)
        final = clone(model).fit(X_tr, y_tr)
        proba = final.predict_proba(X_te)[:, 1]
        results[name] = {
            "label": MODEL_LABELS[name],
            "cv_pr_auc_mean": round(float(np.mean(folds)), 4),
            "cv_pr_auc_std": round(float(np.std(folds)), 4),
            "cv_oof": summarize(y_tr, oof, threshold),
            "test": summarize(y_te, proba, threshold),
            "train_seconds": round(time.time() - t, 1),
        }
        fitted[name], test_proba[name] = final, proba
        print(f"{name:20s} CV PR-AUC {results[name]['cv_pr_auc_mean']:.3f} ± {results[name]['cv_pr_auc_std']:.3f}"
              f" | test PR-AUC {results[name]['test']['pr_auc']:.3f} recall {results[name]['test']['recall']:.3f}"
              f" precision {results[name]['test']['precision']:.3f}")

    servable = [n for n in names if n != "pytorch_mlp"]
    served = max(servable, key=lambda n: results[n]["cv_pr_auc_mean"])
    best_overall = max(names, key=lambda n: results[n]["cv_pr_auc_mean"])

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[served], "name": served, "label": MODEL_LABELS[served],
                 "threshold": results[served]["test"]["threshold"]},
                out_dir / "model.joblib", compress=3)

    import sklearn

    metrics = {
        "project_version": __version__,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fast_mode": fast,
        "dataset": {"name": "AI4I 2020 Predictive Maintenance Dataset (UCI)", "sha256": ds.sha256(),
                    **report.__dict__},
        "split": {"train_rows": int(len(y_tr)), "test_rows": int(len(y_te)),
                  "train_failures": int(y_tr.sum()), "test_failures": int(y_te.sum()), "seed": SEED,
                  "cv_folds": 5},
        "raw_features": RAW_FEATURES,
        "model_features": FEATURE_NAMES,
        "training_ranges": {col: [float(X_tr[:, i + 1].astype(float).min()), float(X_tr[:, i + 1].astype(float).max())]
                            for i, col in enumerate(RAW_FEATURES[1:])},
        "models": results,
        "served_model": served,
        "best_model_overall": best_overall,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__},
        "total_seconds": round(time.time() - t0, 1),
    }
    if fig_dir is not None:
        metrics["figures"] = make_figures(fig_dir, results, test_proba, y_te, fitted[served], served,
                                          X_te)
        metrics["permutation_importance"] = metrics["figures"].pop("permutation_importance")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"served model: {served} | best overall: {best_overall} | {metrics['total_seconds']}s")
    return metrics


def make_figures(fig_dir: Path, results, test_proba, y_te, served_model, served, X_te) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import PrecisionRecallDisplay

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name, proba in test_proba.items():
        PrecisionRecallDisplay.from_predictions(
            y_te, proba, ax=ax, name=results[name]["label"])
    ax.axhline(y_te.mean(), ls="--", c="grey", lw=1, label=f"Always 'failure' ({y_te.mean():.3f})")
    ax.set_xlabel("Recall (share of failures caught)")
    ax.set_ylabel("Precision (share of alarms that are real)")
    ax.set_title("Precision-recall on the held-out test set")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curves.png", dpi=130)
    plt.close(fig)

    cm = results[served]["test"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(grid, cmap="Blues")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > grid.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No failure", "Failure"])
    ax.set_yticks([0, 1], ["No failure", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{results[served]['label']}, test set")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    imp = permutation_importance(served_model, X_te, y_te, scoring="average_precision",
                                 n_repeats=10, random_state=SEED, n_jobs=1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.barh(np.array(RAW_FEATURES)[order], imp.importances_mean[order], xerr=imp.importances_std[order],
            color="#5b3cf5")
    ax.set_xlabel("Drop in test PR-AUC when the column is shuffled")
    ax.set_title("Which inputs the served model relies on")
    fig.tight_layout()
    fig.savefig(fig_dir / "permutation_importance.png", dpi=130)
    plt.close(fig)
    return {"files": ["pr_curves.png", "confusion_matrix.png", "permutation_importance.png"],
            "permutation_importance": {f: round(float(m), 4) for f, m in
                                       zip(RAW_FEATURES, imp.importances_mean)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    ap.add_argument("--figures", default=str(ROOT / "public" / "figures"),
                    help="folder for plots; pass '' to skip plotting")
    ap.add_argument("--fast", action="store_true", help="smaller models, for smoke tests")
    ap.add_argument("--models", nargs="*", help=f"subset of {list(MODEL_LABELS)}")
    a = ap.parse_args()
    run(Path(a.out), Path(a.figures) if a.figures else None, fast=a.fast, models=a.models)


if __name__ == "__main__":
    main()
