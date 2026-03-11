from datetime import datetime

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)

    email = Column(String, unique=True, index=True, nullable=False)

    password = Column(String, nullable=False)


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer)

    patient_name = Column(String)

    vitamin_A = Column(Float)
    vitamin_D = Column(Float)
    glucose = Column(Float)
    iron = Column(Float)

    prediction = Column(Float)
    severity = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)