"""Module for the adult income model training."""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from utils.aws.s3 import S3Buckets


def train_model(features: pd.DataFrame, target: pd.Series,
                bucket_name: str,filename: str,
                folder: str) -> XGBClassifier:
    """
    Train a model using the XGBoost Classifier.

    Args:
        features (pd.DataFrame): The features to train the model on.
        target (pd.Series): The target variable.
        bucket_name (str): The name of the S3 bucket to store the model.
        filename (str): The name of the model file.
        folder (str): The folder to store the model in the S3 bucket.

    Returns:
        XGBClassifier: The trained model.
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


def validation_predictions(features:pd.DataFrame,
                   model:XGBClassifier) -> np.ndarray:
    """Generate predictions on validation data using a trained XGBoost classifier.

    Args:
        features (pd.DataFrame): Input features for the validation set, where each column
            represents a feature and each row is a sample.
        model (XGBClassifier): Trained XGBoost classifier used to make predictions.

    Returns:
        np.ndarray: Array of predicted class labels for the validation set.
    """
    return model.predict(features)


def test_predictions(features:pd.DataFrame,
                   model:XGBClassifier) -> np.ndarray:
    """Generate predictions on test data using a trained XGBoost classifier.

    Args:
        features (pd.DataFrame): Input features for the validation set, where each column
            represents a feature and each row is a sample.
        model (XGBClassifier): Trained XGBoost classifier used to make predictions.

    Returns:
        np.ndarray: Array of predicted class labels for the validation set.
    """
    return model.predict(features)
