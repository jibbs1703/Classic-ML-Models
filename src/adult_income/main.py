"""Main module for the adult income classification project."""
from adult_income import config
from adult_income.etl.extract import run_data_extraction
from adult_income.etl.load import load_dataset
from adult_income.etl.transform import Preprocess
from adult_income.train.train import test_model, train_model, validate_model
from utils.helpers.etl_helpers import resample_training, train_test_val_split

if __name__ == "__main__":
    # Perform ETL on Data
    df = run_data_extraction(config.RAW_BUCKET_NAME, config.RAW_FILE_NAME)

    preprocessor = Preprocess(df=df, numeric_columns=config.NUMERIC_COLUMNS,
                              target=config.TARGET, categorical_columns=config.CATEGORICAL_COLUMNS,
                              bucket_name=config.CLEAN_BUCKET_NAME, folder=config.ARTIFACT_FOLDER)
    df = preprocessor.run_preprocessor()
    
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(
                                                     df=df,target=config.TARGET,split1=0.2,split2=0.3
                                                     )
    X_train, y_train = resample_training(
        X=X_train, y=y_train,categorical_features=config.CATEGORICAL_COLUMNS)
    datasets = {'X_train': X_train,'X_test': X_test,'X_val': X_val,
                  'y_train': y_train,'y_test': y_test,'y_val': y_val}
    for filename,df in datasets.items():
        load_dataset(bucket_name=config.CLEAN_BUCKET_NAME,filename=filename, df=df)
    
    # Train Model, Store Artifacts and Evaluate
    model = train_model(features=X_train, target=y_train,
                bucket_name=config.CLEAN_BUCKET_NAME,
                              filename=config.MODEL_NAME,
                              folder=config.ARTIFACT_FOLDER)
    
    test_model(features=X_test, target=y_test, model=model)
    validate_model(features=X_val, target=y_val, model= model)