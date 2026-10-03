# Lab-1-API

A Python lab for training a Boston Housing linear regression model and exposing predictions through FastAPI, Flask, and Streamlit. FastAPI provides a JSON API with interactive documentation, Flask provides a JSON API and a browser form, and Streamlit provides an interactive prediction interface.

## Repository structure

```text
Lab-1-API/
├── data/housing.csv              # Whitespace-delimited dataset without a header
├── src/
│   ├── data.py                   # Column names, data loading, and train/test split
│   └── model.py                  # Training, evaluation, and model serialization
├── util/log.py                  # Loguru logging to standard output
├── model/model.pkl              # Model used by FastAPI and Streamlit
├── fastapi_lab/main.py          # FastAPI POST /predict endpoint
├── flask_lab/
│   ├── main.py                  # Flask homepage and POST /predict endpoint
│   ├── model/model.pkl          # Separate model used by Flask
│   └── templates/index.html     # Browser form that calls the Flask API
├── streamlit_lab/main.py        # Streamlit form and prediction display
├── environment.yml             # Conda environment, including Python 3.13
└── requirements.txt            # Core Python dependencies
```

The package directories also contain `__init__.py` files. `.vscode/settings.json` selects Conda as the editor's default environment and package manager; `.gitignore` excludes common Python build, cache, and environment files.

## Setup

Run the commands below from the repository root. Python 3.13 is specified in the Conda environment.

### Using Conda

```bash
conda env create -f environment.yml
conda activate dads7305_lab_api
```

### Using a virtual environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead.

`requirements.txt` lists NumPy, pandas, FastAPI, scikit-learn, Joblib, Flask, and Streamlit. `environment.yml` additionally includes Jupyter and Matplotlib. Neither file explicitly lists Loguru or Uvicorn, so the extra installation command is needed for logging and the FastAPI launch command. The API implementations use Pydantic 2 APIs (`ConfigDict` and `model_dump`). Dependencies are not locked to exact versions.

## Train the model

```bash
python -m src.model
```

Training loads the 506 rows and 14 columns in `data/housing.csv`, uses the 13 input columns to predict `MEDV`, and fits scikit-learn's `LinearRegression` without an additional preprocessing pipeline. It uses a shuffled 70% training / 30% test split with `random_state=42`, logs MSE, RMSE, and R² on the test set, and writes the model to `model/model.pkl`, overwriting any existing file there.

## Run the applications

Each application runs independently. Use separate terminals with the environment activated if running them together.

| Application | Command from repository root | Local URL |
| --- | --- | --- |
| FastAPI | `python -m uvicorn fastapi_lab.main:app --reload --port 8000` | `http://127.0.0.1:8000/docs` |
| Flask | `python -m flask_lab.main` | `http://127.0.0.1:8001/` |
| Streamlit | `python -m streamlit run streamlit_lab/main.py` | `http://localhost:8501/` |

FastAPI exposes interactive Swagger UI at `/docs`, ReDoc at `/redoc`, and its OpenAPI schema at `/openapi.json`. It has no application homepage at `/`.

Flask serves the prediction form at `/`. Its launch code binds to `0.0.0.0:8001` with debug mode disabled. The form sends JSON to `/predict` and displays the result in thousands of dollars.

Streamlit predicts directly with the saved model rather than calling either API. Its form includes example values, an **Estimate value** button, and a **Reset example** button. It displays the estimate in dollars by multiplying the model output by 1,000, and reports model or input errors on the page.

## Prediction API

Both APIs accept `POST /predict` with `Content-Type: application/json`. Supply all 13 feature names with the exact capitalization shown below. The server constructs a one-row DataFrame in the model's feature order.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.9,
    "LSTAT": 4.98
  }'
```

For Flask, use the same request with `http://127.0.0.1:8001/predict`.

A successful response is a JSON object containing a numeric `prediction`. For example, `{"prediction": 25.0}` represents an estimated median value of $25,000 in the original dataset's dollars; this illustrates the response format, not the exact result of the request above.

Both API schemas require all fields, reject extra fields, and disallow non-finite numbers. They do not impose feature-specific ranges such as `CHAS` being 0 or 1. FastAPI returns HTTP 422 for schema validation failures. Flask does not define a handler for Pydantic validation errors, so those failures currently produce HTTP 500 rather than a structured validation response. The endpoints accept one observation per request.

## Dataset and features

`data/housing.csv` is parsed as whitespace-delimited numeric data without a header. `src/data.py` assigns these columns in order:

| Column | Meaning used by the application |
| --- | --- |
| `CRIM` | Crime rate |
| `ZN` | Percentage of residential land zoned for large lots |
| `INDUS` | Percentage of non-retail business land |
| `CHAS` | Charles River boundary indicator: 0 or 1 |
| `NOX` | Nitric oxide concentration |
| `RM` | Average rooms per dwelling |
| `AGE` | Percentage of homes built before 1940 |
| `DIS` | Employment distance |
| `RAD` | Highway accessibility index |
| `TAX` | Property tax rate |
| `PTRATIO` | Pupil–teacher ratio |
| `B` | Historical race-derived index |
| `LSTAT` | Historical lower-status population indicator (%) |
| `MEDV` | Target: median home value in thousands of dollars |