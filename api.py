# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
from src.data_processing import preprocess_data

app = FastAPI(title="Telco Churn Prediction API")

# Load model once at startup
model = joblib.load("models/best_rf.pkl")

# --- Define request model ---
class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None

class Customers(BaseModel):
    customers: List[Customer]

# --- Predict endpoint ---
@app.post("/predict")
def predict(data: Customers):
    # Convert to DataFrame
    df = pd.DataFrame([c.dict() for c in data.customers])
    
    # Preprocess
    df_processed = preprocess_data(df)
    X = df_processed.drop(columns=['customerID', 'Churn'], errors='ignore')
    
    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    # Combine results
    df_processed['Churn_Prediction'] = preds
    df_processed['Churn_Probability'] = probs
    
    return df_processed[['customerID', 'Churn_Prediction', 'Churn_Probability']].to_dict(orient="records")
