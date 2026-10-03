import os

import pandas as pd
from sklearn.model_selection import train_test_split

from util.log import log

COLUMNS = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]

def load_data(path="./data/housing.csv"):
    if not os.path.exists(path):
        log.error(f"{path} does not exist.")
        return None

    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=COLUMNS,
        )

        log.info(
            f"Loaded dataset successfully: {df.shape[0]} rows, "
            f"{df.shape[1]} columns."
        )

        return df

    except Exception as e:
        log.exception(f"Failed to load dataset: {e}")
        raise

def split_data(X, y, train=0.7, test=0.3, random_state=42):
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"X should be a pandas DataFrame, not {type(X).__name__}."
        )

    if not isinstance(y, pd.Series):
        raise TypeError(
            f"y should be a pandas Series, not {type(y).__name__}."
        )

    if not (0 <= train <= 1):
        raise ValueError("train must be between 0 and 1.")

    if not (0 <= test <= 1):
        raise ValueError("test must be between 0 and 1.")

    if not abs(train + test - 1.0) < 1e-9:
        raise ValueError("train + test must equal 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train,
        test_size=test,
        random_state=random_state,
        shuffle=True,
    )

    log.debug(
        f"Data split into train and test sets with a ratio of "
        f"{train}:{test}."
    )

    return X_train, X_test, y_train, y_test
