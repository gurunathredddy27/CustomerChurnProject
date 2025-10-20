# src/evaluate.py (precision at top K)
 

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve
from data_processing import load_data
from src.features import prepare_X_y

def precision_at_k(y_true, y_score, k=0.1):
    # k: fraction (e.g., 0.1 means top 10%)
    n = int(len(y_score) * k)
    idx = np.argsort(y_score)[-n:]
    return y_true.iloc[idx].mean()


if __name__ == '__main__':
    df = load_data()
    X, y, preproc = prepare_X_y(df)
    model = joblib.load('models/best_rf.pkl')


    probs = model.predict_proba(X)[:, 1]
    print('AUC:', roc_auc_score(y, probs))
    print('Precision@10%:', precision_at_k(y, probs, k=0.1))