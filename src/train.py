# src/train.py
from .data_processing import preprocess_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "best_rf.pkl")

# Load CSV
CSV_PATH = "data/raw/Telco-Customer-Churn.csv"
df = pd.read_csv(CSV_PATH)

# Preprocess
df = preprocess_data(df)

# Train/test split
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

y_pred_prob = clf.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, y_pred_prob))
print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
