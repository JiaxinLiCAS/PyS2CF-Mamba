"""Hyperspectral classification metrics."""

import numpy as np


def classification_metrics(
    prediction: np.ndarray,
    label_mask: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    valid = (label_mask >= 0) & (label_mask < num_classes)
    encoded = num_classes * label_mask[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    matrix = np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)
    total = matrix.sum()
    if total == 0:
        raise ValueError("The evaluation mask contains no labelled pixels.")

    true_positive = np.diag(matrix).astype(np.float64)
    true_count = matrix.sum(axis=1).astype(np.float64)
    per_class_accuracy = np.divide(
        true_positive,
        true_count,
        out=np.full_like(true_positive, np.nan),
        where=true_count > 0,
    )
    oa = float(true_positive.sum() / total)
    aa = float(np.nanmean(per_class_accuracy))
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    expected = float(np.sum(true_count * predicted_count) / (total**2))
    kappa = float((oa - expected) / (1.0 - expected)) if expected < 1.0 else 0.0
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "per_class_accuracy": per_class_accuracy.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
