# Adult Income Classification Model

This project builds a machine learning model to classify adult income levels (e.g., ≤50K or >50K) based on demographic and employment data. The pipeline includes data processing, model experimentation, training, evaluation, and deployment via a FastAPI application for real-time predictions.

## Data Processing and Preparation

The data processing pipeline prepares the Adult Income dataset (e.g., from the UCI repository) for training machine learning models. The steps are as follows:

- **Data Extraction**: Raw data is extracted from a source (e.g., CSV file or database).
- **Preprocessing**: 
  - Categorical features are encoded (e.g., using one-hot or label encoding).
  - Numerical features are scaled (e.g., using StandardScaler).
  - Missing values are handled (e.g., imputation or removal).
- **Splitting**: The dataset is divided into training (70%), validation (15%), and testing (15%) sets.
- **Resampling**: Techniques like SMOTE or undersampling are applied to address class imbalance in the target variable (income).
- **Storage**: Processed datasets are uploaded to an S3 bucket (`CLEAN_BUCKET_NAME`) for use in training and deployment.

These steps ensure the data is clean, balanced, and ready for model training.

## Model Training

The model training process involves experimenting with multiple algorithms, selecting the best performer, and tuning its parameters. Key aspects include:

- **Experimentation**: Three models were evaluated in an experiments notebook mimicking MLflow-style logging:
  - **Logistic Regression**: A baseline linear model for binary classification.
  - **Random Forest**: An ensemble model to capture non-linear relationships.
  - **XGBoost**: A gradient boosting model for high performance and scalability.
- **Parameter Tuning**: Hyperparameters for each model were tested using grid search or random search in the experiments notebook. Results (e.g., accuracy, F1-score) and parameters were logged in the `experiments/` folder.
- **Model Selection**: XGBoost was chosen as the final model due to its superior performance (e.g., highest F1-score and robustness to overfitting).
- **Training**: The XGBoost model was trained on the preprocessed training set with optimized parameters (e.g., `max_depth`, `learning_rate`, `n_estimators`).
- **Threshold Determination**: A probability threshold (e.g., 0.5) was selected to classify predictions into ≤50K or >50K categories, optimized based on validation set performance.

The trained model, scaler, and encoder are saved as artifacts (e.g., `model.json`, `scaler.joblib`, `encoder.joblib`) and uploaded to S3.

## Model Evaluation

The XGBoost model was evaluated on the test set to assess its performance. Key metrics include:

- **Accuracy**: Proportion of correct predictions.
- **Precision**: Ratio of true positives to predicted positives.
- **Recall**: Ratio of true positives to actual positives.
- **F1-Score**: Harmonic mean of precision and recall, prioritizing balanced performance.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve, measuring discrimination ability.

Evaluation results were logged and visualized (e.g., confusion matrix, ROC curve) in the experiments notebook. The model demonstrated strong predictive power, particularly after addressing class imbalance and tuning.

## Prediction with Trained Model

A simple FastAPI application was developed to serve predictions using the trained XGBoost model. The deployment includes:

- **Artifact Loading**: On startup, the app loads the model, scaler, and encoder from S3 using a custom `S3Client` class.
- **API Endpoint**: A `/predict` endpoint accepts JSON input with feature data (e.g., age, education, occupation) and returns the predicted income category.
  - sample request:
    ```json
    {
      "age": 39,
      "workclass": "Private",
      "education": "Bachelors",
      "occupation": "Exec-managerial",
      ...
    }
    ```
  - sample response:
    ```json
    {
      "prediction": ">50K",
      "probability": 0.78
    }
    ```
- **Preprocessing**: Input data is preprocessed (scaled and encoded) using the loaded scaler and encoder before prediction.
- **Deployment**: The app runs with Uvicorn and can be containerized with Docker for scalability.

The FastAPI app leverages the `@asynccontextmanager` lifespan to manage resource loading and cleanup, ensuring efficient operation.
