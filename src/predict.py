# src/predict.py
import pandas as pd
import joblib
from .data_processing import preprocess_data  # relative import

# Load trained model
model = joblib.load("models/best_rf.pkl")

# Load new data (can be one row or CSV)
df_new = pd.read_csv("data/raw/Telco-Customer-Churn.csv")  # or your new customer data
df_new = preprocess_data(df_new)

# Drop target if exists
X_new = df_new.drop(columns=['customerID', 'Churn'], errors='ignore')

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Add predictions to dataframe
df_new['Churn_Prediction'] = predictions
df_new['Churn_Probability'] = probabilities

print(df_new[['customerID', 'Churn_Prediction', 'Churn_Probability']].head())
