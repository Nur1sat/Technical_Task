from app.training.metrics import compute_classification_metrics


def test_compute_classification_metrics_returns_expected_scores() -> None:
    summary = compute_classification_metrics(
        losses=[0.2, 0.4, 0.6, 0.8],
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
    )

    assert summary.loss == 0.5
    assert summary.accuracy == 0.75
    assert summary.precision == 2 / 3
    assert summary.recall == 1.0
    assert round(summary.f1, 4) == 0.8
    assert summary.confusion_matrix == [[1, 1], [0, 2]]
