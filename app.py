import sys, subprocess
try:
    import joblib  # noqa
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.4.2"])
    import joblib  # noqa
import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page config (no emoji/logo) ----------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ---------- Utils ----------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

def _extract_feature_names(model) -> Optional[List[str]]:
    # Try common places scikit-learn stores feature names
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass
    # Pipelines (sklearn)
    try:
        if hasattr(model, "named_steps"):
            for _, step in model.named_steps.items():
                try:
                    if hasattr(step, "feature_names_in_"):
                        return list(step.feature_names_in_)
                except Exception:
                    pass
                try:
                    if hasattr(step, "transformers_"):
                        cols_all = []
                        for _, trans, cols in step.transformers_:
                            if isinstance(cols, (list, tuple, np.ndarray)):
                                cols_all.extend(list(cols))
                        if cols_all:
                            # unique in order
                            seen, ordered = set(), []
                            for c in cols_all:
                                if isinstance(c, str) and c not in seen:
                                    ordered.append(c); seen.add(c)
                            if ordered:
                                return ordered
                except Exception:
                    pass
    except Exception:
        pass
    return None

def _align_dataframe(df: pd.DataFrame, feature_names: Optional[List[str]]) -> pd.DataFrame:
    if feature_names is None:
        return df
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_names]

def _coerce_types(row_dict: dict):
    out = {}
    for k, v in row_dict.items():
        if isinstance(v, str):
            s = v.strip()
            # Empty
            if s == "":
                out[k] = np.nan
                continue
            # Try numeric
            try:
                if s.replace(".","",1).isdigit():
                    # handle ints & floats
                    out[k] = float(s) if ("." in s) else int(s)
                    continue
            except Exception:
                pass
            # Leave as text
            out[k] = v
        else:
            out[k] = v
    return out

# ---------- Sidebar: model loader ----------
with st.sidebar:
    st.header("Model Loader")
    default_path = "best_price_model.joblib"
    model_path = st.text_input("Path to saved model (.joblib)", value=default_path)
    load_btn = st.button("Load/Reload model", type="primary")

model = None
feature_names: Optional[List[str]] = None
if load_btn or os.path.exists(model_path):
    try:
        model = load_model(model_path)
        feature_names = _extract_feature_names(model)
        st.sidebar.success(f"Loaded model from: {model_path}")
        if feature_names:
            st.sidebar.caption(f"Detected {len(feature_names)} training features.")
            with st.sidebar.expander("Show feature names"):
                st.sidebar.write(feature_names)
        else:
            st.sidebar.info("Could not auto-detect training feature names. We'll build the row using the UI inputs.")
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")

# ---------- Title ----------
st.title("Car Price Predictor")
st.caption("Enter vehicle details and get a predicted price from your trained model.")

# ---------- Defaults for dropdowns ----------
BRANDS = {
    "Volkswagen": ["Amarok", "Polo", "Golf", "Tiguan"],
    "Toyota": ["Hilux", "Corolla", "Fortuner", "Yaris"],
    "Ford": ["Ranger", "Fiesta", "Focus", "Everest"],
    "BMW": ["1 Series", "3 Series", "X3", "X5"],
    "Mercedes-Benz": ["A-Class", "C-Class", "GLA", "GLE"],
}

SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
TRANSM_TYPES = ["Manual", "Automatic"]

# ---------- Single Prediction UI (sliders + dropdowns) ----------
if model is None:
    st.warning("Load your model from the sidebar to begin.")
else:
    st.subheader("Enter Vehicle Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        brand = st.selectbox("Brand", sorted(BRANDS.keys()))
    with c2:
        model_name = st.selectbox("Model", BRANDS.get(brand, []))
    with c3:
        custom_model = st.text_input("Custom model (optional)", value="")
        if custom_model.strip():
            model_name = custom_model.strip()

    car_name = f"{brand} {model_name}".strip()

    c4, c5, c6 = st.columns(3)
    with c4:
        vehicle_age = st.slider("Vehicle age (years)", min_value=0, max_value=30, value=7, step=1)
    with c5:
        km_driven = st.slider("Kilometres driven", min_value=0, max_value=500_000, value=120_000, step=1_000)
    with c6:
        seats = st.slider("Seats", min_value=2, max_value=10, value=5, step=1)

    c7, c8, c9 = st.columns(3)
    with c7:
        mileage = st.slider("Mileage (km/l)", min_value=3.0, max_value=35.0, value=15.0, step=0.1)
    with c8:
        engine = st.slider("Engine size (cc)", min_value=600, max_value=6000, value=2000, step=50)
    with c9:
        max_power = st.slider("Max power (bhp)", min_value=20, max_value=600, value=120, step=5)

    c10, c11, c12 = st.columns(3)
    with c10:
        seller_type = st.selectbox("Seller type", SELLER_TYPES)
    with c11:
        fuel_type = st.selectbox("Fuel type", FUEL_TYPES)
    with c12:
        transmission_type = st.selectbox("Transmission", TRANSM_TYPES)

    # Build a single-row input dict using common Kaggle/Car columns
    row = {
        "car_name": car_name,
        "brand": brand,
        "model": model_name,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }

    # Allow manual editing of raw JSON (optional expert mode)
    with st.expander("Advanced: edit raw JSON used for prediction"):
        st.code(json.dumps(row, indent=2), language="json")

    # Predict
    if st.button("Predict price", type="primary"):
        try:
            row = _coerce_types(row)
            df = pd.DataFrame([row])
            df_aligned = _align_dataframe(df.copy(), feature_names)
            y_pred = model.predict(df_aligned)
            pred_value = float(np.ravel(y_pred)[0])

            st.success(f"Predicted price: **{pred_value:,.2f}**")
            st.caption("Shown in the model's native target units (e.g., ZAR).")

            st.markdown("**Model input (after alignment):**")
            st.dataframe(df_aligned, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
