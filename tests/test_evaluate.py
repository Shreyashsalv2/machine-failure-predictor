import pytest

from mfp.evaluate import best_f1_threshold, summarize


def test_best_f1_threshold_on_toy_data():
    # At 0.35 both positives are caught with one false alarm (F1 = 0.8), the best available.
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.35, 0.8]
    assert best_f1_threshold(y, p) == pytest.approx(0.35)


def test_summarize_counts_and_scores():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.2, 0.7, 0.8, 0.3]
    s = summarize(y, p, threshold=0.5)
    assert s["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)
    assert 0 <= s["pr_auc"] <= 1 and 0 <= s["roc_auc"] <= 1
