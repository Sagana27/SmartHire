"""Shared model evaluation helpers."""

from sklearn.metrics import accuracy_score, classification_report


def classification_metrics(y_true, y_pred) -> dict:
    """Return common classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }
