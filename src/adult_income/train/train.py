"""Main module for the adult income classification project."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from utils.aws.s3 import S3Buckets


def train_model(features: pd.DataFrame, target: pd.Series,
                bucket_name: str,filename: str,
                folder: str) -> XGBClassifier:
    """_summary_

    Args:
        features (pd.DataFrame): _description_
        target (pd.Series): _description_
        bucket_name (str): _description_
        filename (str): _description_
        folder (str): _description_

    Returns:
        XGBClassifier: _description_
    """
    s3_connection = S3Buckets.credentials()
    model = XGBClassifier(colsample_bytree=0.7,learning_rate=0.1,
                          max_depth=20, n_estimators=200,
                          subsample=1.0, random_state=2024)
    model.fit(features,target)
    s3_connection.save_model_to_s3(model=model,
                              bucket_name=bucket_name,
                              folder=folder,
                              model_name=filename)
    return model


def validate_model(features:pd.DataFrame, target:pd.Series,
                   model:XGBClassifier) -> None:
    """_summary_

    Args:
        features (pd.DataFrame): _description_
        target (pd.Series): _description_
        model (XGBClassifier): _description_
    """
    predictions = model.predict(features)
    print("Model Accuracy:", accuracy_score(target, predictions))
    print("Precision:", precision_score(target, predictions))
    print("Recall:", recall_score(target, predictions))
    print("F1 Score:", f1_score(target, predictions))
    print("ROC-AUC:", roc_auc_score(target, predictions))
    print("Confusion Matrix:\n", confusion_matrix(target, predictions))


def test_model(features:pd.DataFrame, target:pd.Series,
                   model:XGBClassifier) -> None:
    """_summary_

    Args:
        features (pd.DataFrame): _description_
        target (pd.Series): _description_
        model (XGBClassifier): _description_
    """
    predictions = model.predict(features)
    print("Model Accuracy:", accuracy_score(target, predictions))
    print("Precision:", precision_score(target, predictions))
    print("Recall:", recall_score(target, predictions))
    print("F1 Score:", f1_score(target, predictions))
    print("ROC-AUC:", roc_auc_score(target, predictions))
    print("Confusion Matrix:\n", confusion_matrix(target, predictions))
