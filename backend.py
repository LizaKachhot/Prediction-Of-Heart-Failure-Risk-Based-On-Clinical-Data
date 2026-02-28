from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

# =====================================================
# INITIALIZE FASTAPI
# =====================================================
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD TRAINED ARTIFACTS
# =====================================================
artifacts = joblib.load("heart_failure_xgboost_project.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["features"]

# =====================================================
# INPUT SCHEMA (MATCHES YOUR STREAMLIT INPUTS)
# =====================================================
class InputData(BaseModel):
    age: int
    sex: str
    bmi: float
    resting_bp: int
    cholesterol: int
    oldpeak: float
    fasting_blood_sugar: str
    exercise_angina: str
    diabetes: str
    num_major_vessels: str
    chest_pain_type: str
    rest_ecg: str
    slope: str
    thalassemia: str
    smoking_status: str

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def binary_or_nan(val):
    if val == "yes":
        return 1
    if val == "no":
        return 0
    return np.nan

def cat_or_nan(val):
    return val.lower() if val != "Unknown" else np.nan

# ORDINAL ENCODING FOR SLOPE (MATCHES TRAINING)
slope_map = {
    "upsloping": 0,
    "flat": 1,
    "downsloping": 2
}

# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
def predict(data: InputData):

    # -------------------------
    # Build input dataframe
    # -------------------------
    input_df = pd.DataFrame({
        "age": [data.age],
        "sex": [1 if data.sex.lower() == "male" else 0],
        "resting_bp": [data.resting_bp],
        "cholesterol": [data.cholesterol],
        "fasting_blood_sugar": [binary_or_nan(data.fasting_blood_sugar)],
        "exercise_angina": [binary_or_nan(data.exercise_angina)],
        "oldpeak": [data.oldpeak],
        "num_major_vessels": [
            float(data.num_major_vessels)
            if data.num_major_vessels != "Unknown"
            else np.nan
        ],
        "bmi": [data.bmi],
        "diabetes": [binary_or_nan(data.diabetes)],
        "chest_pain_type": [cat_or_nan(data.chest_pain_type)],
        "rest_ecg": [cat_or_nan(data.rest_ecg)],
        "slope": [slope_map.get(data.slope.lower(), np.nan)],
        "thalassemia": [cat_or_nan(data.thalassemia)],
        "smoking_status": [data.smoking_status]
    })

    # -------------------------
    # One-hot encode (EXCEPT slope)
    # -------------------------
    input_df = pd.get_dummies(
        input_df,
        columns=[
            "chest_pain_type",
            "rest_ecg",
            "thalassemia",
            "smoking_status"
        ],
        drop_first=False
    )

    # -------------------------
    # Align with training columns
    # -------------------------
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # -------------------------
    # Apply scaling
    # -------------------------
    input_df[scaler.feature_names_in_] = scaler.transform(
        input_df[scaler.feature_names_in_]
    )

    # -------------------------
    # Predict
    # -------------------------
    probability = float(model.predict_proba(input_df)[0][1] * 100)
    prediction = int(model.predict(input_df)[0])

    return {
        "risk_percentage": round(probability, 2),
        "risk_class": prediction
    }