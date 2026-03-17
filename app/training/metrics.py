from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(slots=True)
class MetricSummary:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, float | int | list[list[int]]]:
        return asdict(self)


def compute_classification_metrics(
    losses: list[float],
    y_true: list[int],
    y_pred: list[int],
) -> MetricSummary:
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)

    if truth.size == 0:
        raise ValueError("No targets provided for metric computation")

    tp = int(np.sum((truth == 1) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))

    accuracy = float((tp + tn) / truth.size)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if precision + recall else 0.0
    loss = float(np.mean(losses)) if losses else 0.0

    return MetricSummary(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        support=int(truth.size),
        confusion_matrix=[[tn, fp], [fn, tp]],
    )
