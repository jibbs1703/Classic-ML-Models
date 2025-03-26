import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.aws.s3 import S3Buckets

load_dotenv()
s3_client = S3Buckets.credentials()

app = FastAPI(
    title="Adult Income Classifier",
    description="Predict income level (<=50K or >50K) based on user inputs.",
    version="1.0.0"
)


class IncomeInput(BaseModel):
    age: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int
    native_country: str
