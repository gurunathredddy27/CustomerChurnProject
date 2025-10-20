# src/explain.py
 

import joblib
import shap
import pandas as pd


from data_processing import load_data
from src.features import prepare_X_y




if __name__ == '__main__':
    df = load_data()
    X, y, preproc = prepare_X_y(df)
    model = joblib.load('models/best_rf.pkl')


    # Use a small subset for speed
    X_small = X.sample(n=500, random_state=42)
    explainer = shap.Explainer(model.named_steps['clf'], model.named_steps['preproc'].transform(X_small))
    shap_values = explainer(model.named_steps['preproc'].transform(X_small))
    shap.summary_plot(shap_values, feature_names=None) # run inside notebook