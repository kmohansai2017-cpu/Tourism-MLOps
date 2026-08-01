"""
train.py
Model Training and Registration Script:
1. Loads the train and test splits produced by the previous job
2. Defines the model and a hyperparameter grid, and tunes it
3. Logs the parameters and metrics to MLflow for experiment tracking
4. Evaluates the best model
5. Saves the best model into tourism_project/deployment/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import os

# Paths
MODEL_OUTPUT_PATH = "tourism_project/deployment/model.pkl"

def main():
    # Load train and test splits
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").values.ravel()
    y_test = pd.read_csv("ytest.csv").values.ravel()

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Identify column types
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Create column transformer for preprocessing
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    )

    # Define XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )

    # Create pipeline
    pipeline = make_pipeline(preprocessor, xgb_clf)

    # Define hyperparameter grid
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.01, 0.1],
    }

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Tourism_Package_Prediction")

    # Start MLflow run
    with mlflow.start_run(run_name="XGBoost_GridSearch"):
        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )

        print("\nStarting GridSearchCV...")
        grid_search.fit(X_train, y_train)

        # Best parameters
        best_params = grid_search.best_params_
        print(f"\nBest Parameters: {best_params}")
        print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

        # Log parameters to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)

        # Evaluate on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Log test metrics to MLflow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)

        # Log model to MLflow
        mlflow.sklearn.log_model(best_model, "xgboost_model")

        # Save model locally for deployment
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_OUTPUT_PATH)
        print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")

    print("\nModel training and registration complete.")

if __name__ == "__main__":
    main()
