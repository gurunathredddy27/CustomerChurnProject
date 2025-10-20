# app.py
import streamlit as st
import pandas as pd
import joblib
from src.data_processing import preprocess_data

st.title("Telco Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload customer CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)

    model = joblib.load("models/best_rf.pkl")
    X = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    df['Churn_Prediction'] = model.predict(X)
    df['Churn_Probability'] = model.predict_proba(X)[:, 1]

    st.write(df[['customerID', 'Churn_Prediction', 'Churn_Probability']])
