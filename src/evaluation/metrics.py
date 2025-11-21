import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)


def calculate_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Returns a complete set of metrics for classification models.
    """
    metrics = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro"))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro"))
    metrics["classification_report"] = classification_report(y_true, y_pred)

    return metrics
