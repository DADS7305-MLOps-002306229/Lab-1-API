from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from src.model import train_model
from util.log import log


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model" / "model.pkl"

FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]

try:
    if MODEL_PATH.exists():
        log.info("Loading model.")
        model = joblib.load(MODEL_PATH)
    else:
        log.info("Model not found. Training model.")
        model = train_model()

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

    log.info("Model ready.")

except Exception:
    log.exception("Could not initialize model.")
    raise


class Parameters(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float


app = FastAPI()

@app.post("/predict")
def predict(data: Parameters):
    df = pd.DataFrame([data.model_dump()], columns=FEATURES)
    prediction = model.predict(df)

    return {"prediction": float(prediction[0])}
