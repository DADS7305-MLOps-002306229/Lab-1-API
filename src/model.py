from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.data import load_data, split_data
from util.log import log

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "housing.csv"
MODEL_PATH = ROOT_DIR / "model" / "model.pkl"

TARGET_COLUMN = "MEDV"

def train_model():
    log.info("Starting model training.")

    # Load dataset
    df = load_data(DATA_PATH)

    if df is None:
        log.error("Failed to load dataset.")
        raise RuntimeError("Dataset could not be loaded.")

    log.debug(f"Dataset loaded with shape: {df.shape}")

    # Check target column
    if TARGET_COLUMN not in df.columns:
        log.error(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    log.debug(
        f"Prepared features X={X.shape}, target y={y.shape}"
    )

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        train=0.7,
        test=0.3,
        random_state=42,
    )

    log.debug(
        f"Training samples: {len(X_train)}, "
        f"Testing samples: {len(X_test)}"
    )

    model = LinearRegression()

    log.info("Training Linear Regression model.")

    model.fit(X_train, y_train)

    log.info("Model training completed.")

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    log.info(f"MSE: {mse:.4f}")
    log.info(f"RMSE: {rmse:.4f}")
    log.info(f"R2 Score: {r2:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    
    log.info(f"Model saved to: {MODEL_PATH}")
    
    return model


if __name__ == "__main__":
    train_model()
