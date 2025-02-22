"""module for data transformation abd preprocessing."""
from dataclasses import dataclass, field

from category_encoders import TargetEncoder
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.aws.s3 import S3Buckets


@dataclass
class Preprocess:
    bucket_name: str
    folder: str
    df: DataFrame = None
    numeric_columns: list[str] = field(default_factory=list)
    target: str = ''
    categorical_columns: list[str] = field(default_factory=list)

    def scale_numeric(self) -> None:
        s3_connection = S3Buckets.credentials()
        # Scaling the numerical columns
        scaler = MinMaxScaler()
        self.df[self.numeric_columns] = scaler.fit_transform(self.df[self.numeric_columns])
        # Store Scaler Artifact for Prediction
        s3_connection.save_model_to_s3(model=scaler,
                                    bucket_name=self.bucket_name,
                                    folder=self.folder,
                                    model_name="adult_model_scaler")
    
    def encode_target(self) -> None:
        # Label Encode the Target
        le = LabelEncoder()
        self.df[self.target] = le.fit_transform(self.df[self.target])

    def encode_categorical(self) -> None:
        s3_connection = S3Buckets.credentials()
        # Target Encoding the Categorical Columns
        encoder = TargetEncoder()
        self.df[self.categorical_columns] = encoder.fit_transform(
            self.df[self.categorical_columns], self.df[self.target])        
        s3_connection.save_model_to_s3(model=encoder,
                                       bucket_name=self.bucket_name,
                                       folder=self.folder,
                                        model_name="adult_model_encoder")
        
    def run_preprocessor(self) -> DataFrame:
        self.scale_numeric()
        self.encode_target()
        self.encode_categorical()
        return self.df
