"""Module to evaluate the model."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_results(target: pd.Series, 
                     predictions: pd.Series) -> dict:
    """
    Generates a list of evaluation metrics for the model.

    Args:
        target (pd.Series): the true target values
        predictions (pd.Series): the predicted target values

    Returns:
        dict: a dictionary containing the evaluation metrics

    """

    return {"Model Accuracy": f"{accuracy_score(target, predictions) * 100:.2f}%",
        "Precision": f"{precision_score(target, predictions) * 100:.2f}%",
        "Recall": f"{recall_score(target, predictions) * 100:.2f}%",
        "F1 Score": f"{f1_score(target, predictions) * 100:.2f}%",
        "ROC-AUC": f"{roc_auc_score(target, predictions) * 100:.2f}%",
        "Confusion Matrix": confusion_matrix(target, predictions)}
