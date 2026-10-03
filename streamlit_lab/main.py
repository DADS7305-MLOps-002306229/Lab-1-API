"""Save as streamlit_lab/main.py; run from the project root."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model" / "model.pkl"
FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
DEFAULTS = dict(zip(FEATURES, [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]))
LABELS = {
    "CRIM": "Crime rate", "ZN": "Large-lot zoning (%)", "INDUS": "Non-retail business land (%)",
    "CHAS": "Charles River boundary", "NOX": "Nitric oxide concentration", "RM": "Average rooms",
    "AGE": "Homes built before 1940 (%)", "DIS": "Employment distance", "RAD": "Highway accessibility index",
    "TAX": "Property tax rate", "PTRATIO": "Pupil–teacher ratio", "B": "Historical race-derived B index",
    "LSTAT": "Historical lower-status indicator (%)",
}

st.set_page_config(page_title="Housing prediction", layout="centered")
st.title("Boston Housing Prediction")
st.caption("Enter neighborhood characteristics and estimate median home value.")


def reset_inputs():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("result", None)
    st.session_state["reset_notice"] = True
    print("[Housing] Reset example clicked", flush=True)


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

st.button("Reset example", on_click=reset_inputs)
if st.session_state.pop("reset_notice", False):
    st.success("Inputs reset to the example values.")

with st.form("prediction_form"):
    columns = st.columns(2)
    inputs = {}
    for index, key in enumerate(FEATURES):
        with columns[index % 2]:
            label = f"{LABELS[key]} ({key})"
            if key == "CHAS":
                inputs[key] = st.selectbox(label, [0, 1], key=key,
                    format_func=lambda value: "Borders river" if value else "Does not border river")
            elif key == "RAD":
                inputs[key] = st.number_input(label, min_value=1, step=1, key=key)
            else:
                maximum = 100.0 if key in {"ZN", "INDUS", "AGE", "LSTAT"} else 396.9 if key == "B" else None
                inputs[key] = st.number_input(label, min_value=0.0, max_value=maximum,
                    format="%.5f" if key == "CRIM" else "%.3f", key=key)
    submitted = st.form_submit_button("Estimate value", type="primary")

# Handle the submission directly below the form so feedback is easy to find.
if submitted:
    print("[Housing] Estimate value clicked", flush=True)
    st.session_state.pop("result", None)
    try:
        with st.spinner("Loading model and calculating prediction…"):
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
            feature_order = list(getattr(model, "feature_names_in_", FEATURES))
            if len(feature_order) != 13 or set(feature_order) != set(FEATURES):
                raise ValueError("Saved model features do not match these 13 inputs.")
            frame = pd.DataFrame([inputs], columns=feature_order)
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError("All input values must be finite.")
            result = float(np.asarray(model.predict(frame)).reshape(-1)[0])
            if not np.isfinite(result):
                raise ValueError("Model returned a non-finite prediction.")
            st.session_state["result"] = result
        print(f"[Housing] Prediction completed: {result}", flush=True)
    except Exception as exc:
        print(f"[Housing] Prediction failed: {exc}", flush=True)
        st.error(f"Prediction failed: {exc}")

if "result" in st.session_state:
    result = st.session_state["result"]
    st.success("Prediction completed.")
    st.metric("Estimated median home value", f"${result * 1000:,.0f}")
    st.caption(f"MEDV: {result:.3f} × $1,000. Result reflects the last submitted inputs.")
    if result < 0:
        st.warning("The model returned a negative value, which is not a meaningful home valuation.")

st.caption("Historical dataset dollars; for learning, not current property valuation.")

