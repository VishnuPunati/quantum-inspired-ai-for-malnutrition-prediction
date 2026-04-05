import os
import joblib
import pandas as pd

from fastapi import Depends
from sqlalchemy.orm import Session
from database import get_db
from models import Report

from auth_dependency import get_current_user
from models import Report, User

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved")

QUANTUM_MODEL_PATH = os.path.join(MODEL_DIR, "quantum_model.pkl")
CLASSICAL_MODEL_PATH = os.path.join(MODEL_DIR, "classical_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

if not os.path.exists(QUANTUM_MODEL_PATH):
    raise FileNotFoundError(f"Quantum model not found at {QUANTUM_MODEL_PATH}")

if not os.path.exists(CLASSICAL_MODEL_PATH):
    raise FileNotFoundError(f"Classical model not found at {CLASSICAL_MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

try:
    quantum_model = joblib.load(QUANTUM_MODEL_PATH)
except Exception:
    quantum_model = None
classical_model = joblib.load(CLASSICAL_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

USE_CLASSICAL = True


def predict_patient(vitamin_A: float, vitamin_D: float, glucose: float, iron: float):

    patient = pd.DataFrame(
        [[vitamin_A, vitamin_D, glucose, iron]],
        columns=["vitamin_A", "vitamin_D", "glucose", "iron"]
    )

    patient_scaled = scaler.transform(patient)

    if USE_CLASSICAL:
        prediction = classical_model.predict(patient_scaled)[0]
    else:
        prediction = quantum_model.predict(patient_scaled)[0]

    message = (
        "Malnutrition Risk Detected"
        if prediction == 1
        else "Normal Nutrition Status"
    )

    return {
        "prediction": int(prediction),
        "message": message
    }
