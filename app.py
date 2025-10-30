# =========================
# Car Price Predictor (Streamlit) — single prediction
# =========================

import sys, subprocess
# Safety net: ensure joblib exists on first boot
try:
    import joblib  # noqa: F401
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.4.2"])
    import joblib  # noqa: F401

import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ---------- Utils ----------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

def _extract_feature_names(model) -> Optional[List[str]]:
    # scikit-learn models
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # Pipelines / ColumnTransformers
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
                        for _, _trans, cols in step.transformers_:
                            if isinstance(cols, (list, tuple, np.ndarray)):
                                cols_all.extend(list(cols))
                        if cols_all:
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
            if s == "":
                out[k] = np.nan
                continue
            # numeric (int/float)
            try:
                # allow one dot and an optional minus for negatives
                sn = s.lstrip("-")
                if sn.replace(".", "", 1).isdigit():
                    out[k] = float(s) if "." in sn else int(s)
                    continue
            except Exception:
                pass
            out[k] = v
        else:
            out[k] = v
    return out

def _resolve_model_path(path: str) -> Optional[str]:
    """Try explicit path, fallbacks, or download if URL."""
    import urllib.request, glob

    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
        local_name = os.path.basename(path.split("?")[0]) or "model.joblib"
        try:
            urllib.request.urlretrieve(path, local_name)
            return local_name if os.path.exists(local_name) else None
        except Exception:
            return None

    candidates = [path, "best_price_model_small.joblib", "best_price_model.joblib"]
    for p in candidates:
        if p and os.path.exists(p):
            return p

    found = sorted(glob.glob("*.joblib"))
    if found:
        st.sidebar.info(f"Found model files in repo: {found}")
    return None

# ---------- Sidebar: model loader ----------
with st.sidebar:
    st.header("Model Loader")
    # prefer the compressed model present in your repo
    default_path = "best_price_model_small.joblib"
    model_path = st.text_input("Path to saved model (.joblib)", value=default_path)
    load_btn = st.button("Load/Reload model", type="primary")

model = None
feature_names: Optional[List[str]] = None

resolved = _resolve_model_path(model_path)
if load_btn or resolved:
    if resolved is None:
        st.sidebar.error(
            "Could not find a model file. Tried:\n"
            f"• {model_path}\n• best_price_model_small.joblib\n• best_price_model.joblib"
        )
    else:
        try:
            model = load_model(resolved)
            feature_names = _extract_feature_names(model)
            st.sidebar.success(f"Loaded model from: {resolved}")
            if feature_names:
                st.sidebar.caption(f"Detected {len(feature_names)} training features.")
                with st.sidebar.expander("Show feature names"):
                    st.sidebar.write(feature_names)
            else:
                st.sidebar.info("Could not auto-detect training feature names. Inputs will still work.")
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
    "Other": [],
}
SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
TRANSM_TYPES = ["Manual", "Automatic"]

# ---------- Single Prediction UI ----------
if model is None:
    st.warning("Load your model from the sidebar to begin.")
else:
    st.subheader("Enter Vehicle Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        brand = st.selectbox("Brand", sorted(BRANDS.keys()), index=sorted(BRANDS.keys()).index("Mercedes-Benz") if "Mercedes-Benz" in BRANDS else 0)
    with c2:
        model_opts = BRANDS.get(brand, [])
        model_name = st.selectbox("Model", model_opts) if model_opts else ""
    with c3:
        custom_model = st.text_input("Custom model (optional)", value="")
        if custom_model.strip():
            model_name = custom_model.strip()

    car_name = f"{brand} {model_name}".strip()

    c4, c5, c6 = st.columns(3)
    with c4:
        vehicle_age = st.slider("Vehicle age (years)", 0, 30, 7, 1)
    with c5:
        km_driven = st.slider("Kilometres driven", 0, 500_000, 120_000, 1_000)
    with c6:
        seats = st.slider("Seats", 2, 10, 5, 1)

    c7, c8, c9 = st.columns(3)
    with c7:
        mileage = st.slider("Mileage (km/l)", 3.0, 35.0, 15.0, 0.1)
    with c8:
        engine = st.slider("Engine size (cc)", 600, 6000, 2000, 50)
    with c9:
        max_power = st.slider("Max power (bhp)", 20, 600, 120, 5)

    c10, c11, c12 = st.columns(3)
    with c10:
        seller_type = st.selectbox("Seller type", SELLER_TYPES)
    with c11:
        fuel_type = st.selectbox("Fuel type", FUEL_TYPES)
    with c12:
        transmission_type = st.selectbox("Transmission", TRANSM_TYPES)

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

    with st.expander("Advanced: edit raw JSON used for prediction"):
        example = json.dumps(row, indent=2)
        st.code(example, language="json")

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
