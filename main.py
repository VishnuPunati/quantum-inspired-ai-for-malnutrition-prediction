from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, engine
from models import Base, Report, User

from model.predict import predict_patient

from auth import hash_password, verify_password, create_access_token
from auth_dependency import get_current_user

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Base.metadata.create_all(bind=engine)


class Patient(BaseModel):
    patient_name: str
    vitamin_A: float
    vitamin_D: float
    glucose: float
    iron: float


@app.get("/")
def home():
    return {"message": "Quantum Care Backend Running"}



@app.post("/register")
def register(name: str, email: str, password: str, db: Session = Depends(get_db)):

    existing_user = db.query(User).filter(User.email == email).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = hash_password(password)

    user = User(
        name=name,
        email=email,
        password=hashed_password
    )

    db.add(user)
    db.commit()

    return {"message": "User registered successfully"}


@app.post("/login")
def login(email: str, password: str, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid password")

    token = create_access_token({"user_id": user.id})

    return {
        "access_token": token,
        "token_type": "bearer"
    }


@app.get("/profile")
def get_profile(
    current_user: User = Depends(get_current_user)
):
    return {
        "name": current_user.name,
        "email": current_user.email,
        "account_type": "Patient",
        "status": "Active"
    }


@app.post("/predict")
def predict(
    data: Patient,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    result = predict_patient(
        data.vitamin_A,
        data.vitamin_D,
        data.glucose,
        data.iron
    )

    prediction_value = result["prediction"]

    # Severity calculation
    severity_score = sum([
        data.vitamin_A < 20,
        data.vitamin_D < 15,
        data.iron < 10,
        data.glucose > 120
    ])

    severity = (
        "LOW" if severity_score == 1 else
        "MEDIUM" if severity_score == 2 else
        "HIGH" if severity_score >= 3 else
        "NORMAL"
    )


    recommendations = []

    if data.vitamin_A < 20:
        recommendations.append("Increase Vitamin A foods (carrots, spinach, milk)")

    if data.vitamin_D < 15:
        recommendations.append("Increase Vitamin D (sunlight, eggs, fortified milk)")

    if data.iron < 10:
        recommendations.append("Increase Iron foods (spinach, lentils, jaggery)")

    if data.glucose > 120:
        recommendations.append("Reduce sugar and refined carbohydrates")

    if not recommendations:
        recommendations.append("Maintain a balanced nutritious diet")

    report = Report(
        user_id=current_user.id,
        patient_name=data.patient_name,
        vitamin_A=data.vitamin_A,
        vitamin_D=data.vitamin_D,
        glucose=data.glucose,
        iron=data.iron,
        prediction=prediction_value,
        severity=severity
    )

    db.add(report)
    db.commit()
    db.refresh(report)

    return {
        "prediction": prediction_value,
        "severity": severity,
        "recommendations": recommendations
    }


@app.get("/history")
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    reports = (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .order_by(Report.created_at.desc())
        .all()
    )

    history = []

    for r in reports:
        history.append({
            "patient_name": r.patient_name,
            "vitamin_A": r.vitamin_A,
            "vitamin_D": r.vitamin_D,
            "glucose": r.glucose,
            "iron": r.iron,
            "prediction": r.prediction,
            "severity": r.severity,
            "created_at": r.created_at
        })

    return history