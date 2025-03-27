"""FastAPI application for predicting income level based on user inputs."""
import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from models import IncomeInput

from adult_income import config
from utils.aws.s3 import S3Buckets

logger = logging.getLogger(__name__)

load_dotenv()
s3_client = S3Buckets.credentials()

ml_models ={}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application and loading resources from S3...")
    ml_models["model"] = s3_client.load_model_from_s3(
        bucket_name=config.CLEAN_BUCKET_NAME,
        model_name=config.MODEL_NAME,
        folder=config.ARTIFACT_FOLDER
    )
    ml_models["scaler"] = s3_client.load_model_from_s3(
        bucket_name=config.CLEAN_BUCKET_NAME,
        model_name=config.SCALER_NAME,
        folder=config.ARTIFACT_FOLDER
    )
    ml_models["encoder"] = s3_client.load_model_from_s3(
        bucket_name=config.CLEAN_BUCKET_NAME,
        model_name=config.ENCODER_NAME,
        folder=config.ARTIFACT_FOLDER
    )
    yield
    logger.info("Shutting down application and clearing resources...")
    ml_models.clear()


app = FastAPI(
    title="Adult Income Classifier",
    description="Predict income level (<=50K or >50K) based on user inputs.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Check if the app and resources are ready."""
    if ml_models["model"] is None and ml_models["scaler"] is None and ml_models["encoder"] is None:
        raise HTTPException(status_code=503, detail="One or More Resources not loaded.")
    return {"status": "healthy", "message": "model, scaler, and encoder are loaded."}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Adult Income Classifier API!"}


@app.post("/predict", response_model=dict)
async def predict_income(input_data: IncomeInput):
    """Predict income classification based on user inputs.

    Args:
        input_data (IncomeInput): User-provided features.

    Returns:
        dict: Prediction result with income class and probability.
    """
    if ml_models["model"] is None or ml_models["scaler"] is None or ml_models["encoder"] is None:
        raise HTTPException(status_code=503, detail="One or More Resources not loaded.")

    input_df = pd.DataFrame([input_data.model_dump()])
    input_df = input_df[config.NUMERIC_COLUMNS + config.CATEGORICAL_COLUMNS]

    input_df[config.NUMERIC_COLUMNS] = ml_models["scaler"].transform(
        input_df[config.NUMERIC_COLUMNS])
    input_df[config.CATEGORICAL_COLUMNS] = ml_models["encoder"].transform(
        input_df[config.CATEGORICAL_COLUMNS])

    prediction = ml_models["model"].predict(input_df)
    probability = ml_models["model"].predict_proba(input_df)

    return {"prediction": prediction[0], "probability": np.max(probability)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)