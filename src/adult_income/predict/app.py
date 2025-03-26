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
from src.utils.aws.s3 import S3Buckets

logger = logging.getLogger(__name__)

load_dotenv()
s3_client = S3Buckets.credentials()

app = FastAPI(
    title="Adult Income Classifier",
    description="Predict income level (<=50K or >50K) based on user inputs.",
    version="1.0.0"
)

model = None
scaler = None
encoder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events for the FastAPI app."""
    # Startup: Load resources from S3
    global model, scaler, encoder
    try:
        logger.info("Loading resources from S3 on startup...")
        model = s3_client.load_model_from_s3(
            bucket_name=config.CLEAN_BUCKET_NAME,
            object_key=config.MODEL_NAME,
            folder=config.ARTIFACT_FOLDER
        )
        scaler = s3_client.load_model_from_s3(
            bucket_name=config.CLEAN_BUCKET_NAME,
            object_key=config.SCALER_NAME,
            folder=config.ARTIFACT_FOLDER
        )
        encoder = s3_client.load_model_from_s3(
            bucket_name=config.CLEAN_BUCKET_NAME,
            object_key=config.ENCODER_NAME,
            folder=config.ARTIFACT_FOLDER
        )
        logger.info("Resources loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load resources from S3: {str(e)}")
        raise
    yield
    logger.info("Shutting down app...")


app.lifespan = lifespan


@app.get("/health")
async def health_check():
    """Check if the app and resources are ready."""
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="One or More Resources not loaded.")
    return {"status": "healthy", "message": "Model, scaler, and encoder are loaded."}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Adult Income Classifier API!"}


@app.post("/predict/", response_model=dict)
async def predict_income(input_data: IncomeInput):
    """Predict income classification based on user inputs.

    Args:
        input_data (IncomeInput): User-provided features.

    Returns:
        dict: Prediction result with income class and probability.
    """
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=503, detail="One or More Resources not loaded.")

    input_df = pd.DataFrame([input_data.dict()])
    input_df = input_df[config.NUMERIC_COLUMNS + config.CATEGORICAL_COLUMNS]

    input_df[config.NUMERIC_COLUMNS] = scaler.transform(input_df[config.NUMERIC_COLUMNS])
    input_df[config.CATEGORICAL_COLUMNS] = encoder.transform(input_df[config.CATEGORICAL_COLUMNS])

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    return {"prediction": prediction[0], "probability": np.max(probability)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)